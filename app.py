import numpy as np
from flask import Flask, request, render_template, jsonify
from io import BytesIO
import pymysql.cursors
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import AzureOpenAI
import annoy
import csv
from bs4 import BeautifulSoup
import pytesseract
from pdf2image import convert_from_bytes
import chardet
import os
import re
import json
import tempfile

# Set the environment variable to disable parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
app = Flask(__name__)

# MySQL Configuration
connection = pymysql.connect(
    host='localhost',
    user='root',
    password='',  # Enter password
    database='docs_chat_bot',
    cursorclass=pymysql.cursors.DictCursor
)

# Azure OpenAI Configuration
azure_openai_endpoint = ''  # Enter endpoint
azure_openai_key = ''  # Enter password

client = AzureOpenAI(
    api_version="", # Enter api version
    azure_endpoint=azure_openai_endpoint,
    api_key=azure_openai_key
)

# Global variables for Annoy index and chunk text mapping
chunk_id_to_text = {}
annoy_index = None
dimension = None

@app.route('/')
def index():
    """
    Route for the home page. Displays the list of projects and the number of files in each project.
    """
    query = """
    SELECT projects.id, projects.name, COUNT(files.id) as file_count
    FROM projects
    LEFT JOIN files ON projects.id = files.project_id
    GROUP BY projects.id, projects.name
    """
    projects = execute_query(query)
    return render_template('index.html', projects=projects)

@app.route('/add_project', methods=['POST'])
def add_project():
    """
    Route to add a new project. Checks if the project already exists.
    """
    project_name = request.form['project_name']

    existing_project = execute_query("SELECT id FROM projects WHERE name = %s", (project_name,))
    if existing_project:
        return jsonify({"message": "Project already exists"}), 400

    try:
        project_id = execute_insert("INSERT INTO projects (name) VALUES (%s)", (project_name,))
        return jsonify({"message": "Project created successfully", "project_id": project_id}), 200
    except Exception as e:
        return jsonify({"message": f"Failed to create project: {str(e)}"}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Route to upload a file. Supports PDF, HTML, and CSV file types.
    Extracts text from the file, processes it into chunks, and creates embeddings for each chunk.
    Stores the chunks and embeddings in the database and builds an Annoy index.
    """
    if 'file' not in request.files or 'project_name' not in request.form:
        return jsonify({"message": "No file or project name provided."}), 400

    file = request.files['file']
    project_name = request.form['project_name']

    if file.filename == '':
        return jsonify({"message": "No selected file."}), 400

    file_extension = file.filename.split('.')[-1].lower()

    if file_extension not in ['pdf', 'html', 'csv']:
        return jsonify({"message": "Unsupported file type."}), 400

    file_content = file.read()
    file_stream = BytesIO(file_content)

    text = extract_text(file_stream, file_extension)
    if not text:
        return jsonify({"message": "Failed to extract text from file."}), 500

    project_id = execute_query("SELECT id FROM projects WHERE name = %s", (project_name,))
    if not project_id:
        return jsonify({"message": f"Project '{project_name}' does not exist."}), 404

    try:
        # Insert the file and get the file_id
        file_id = execute_insert("INSERT INTO files (project_id, filename, text) VALUES (%s, %s, %s)",
                                 (project_id[0]['id'], file.filename, text))

        # Process text into chunks and create embeddings
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(text)
        chunk_id_to_text = {}
        dimension = 0  # Initialize dimension
        annoy_index = annoy.AnnoyIndex(dimension, 'angular')  # Initialize with dummy dimension

        chunk_embeddings = []
        for chunk in chunks:
            chunk_embedding = get_azure_openai_embedding(chunk)
            chunk_embedding_np = np.array(chunk_embedding).astype('float32')

            if dimension == 0:
                dimension = len(chunk_embedding)
                annoy_index = annoy.AnnoyIndex(dimension, 'angular')

            chunk_id = len(chunk_id_to_text)  # Use the length of chunk_id_to_text as a unique id
            chunk_id_to_text[chunk_id] = chunk
            chunk_embeddings.append((chunk_id, chunk_embedding_np))

            # Insert chunk and its embedding into the database
            execute_insert("INSERT INTO chunks (file_id, chunk_text, chunk_embedding) VALUES (%s, %s, %s)",
                           (file_id, chunk, json.dumps(chunk_embedding_np.tolist())))

            annoy_index.add_item(chunk_id, chunk_embedding_np)

        annoy_index.build(10)

        # Serialize the Annoy index
        annoy_index_path = f"/tmp/annoy_index_{file_id}.ann"
        annoy_index.save(annoy_index_path)
        with open(annoy_index_path, 'rb') as f:
            annoy_index_data = f.read()

        # Insert the Annoy index into the database
        execute_insert("INSERT INTO annoy_indexes (file_id, index_blob) VALUES (%s, %s)",
                       (file_id, annoy_index_data))

        return jsonify({"message": "File uploaded successfully"}), 200
    except Exception as e:
        error_message = f"Failed to save file: {str(e)}"
        print(error_message)  # Log error to console for debugging
        return jsonify({"message": error_message}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """
    Route to handle chat interactions. Finds the most relevant chunks based on the question using Annoy index.
    Calls Azure OpenAI API to generate a response based on the context and question.
    """
    global chunk_id_to_text, annoy_index, dimension

    try:
        question = request.form['question']
        project_name = request.form['project_name']
        selected_files = request.form.getlist('files')

        project_id = execute_query("SELECT id FROM projects WHERE name = %s", (project_name,))
        if not project_id:
            return jsonify({"message": f"Project '{project_name}' does not exist."}), 404

        if not selected_files:
            return jsonify({"message": "No files selected."}), 400

        chunk_id_to_text = {}

        # Load embeddings and build chunk_id_to_text mapping
        file_ids = execute_query("SELECT id FROM files WHERE project_id = %s AND filename IN %s",
                                 (project_id[0]['id'], selected_files))
        file_ids = [file['id'] for file in file_ids]

        for file_id in file_ids:
            chunks = execute_query("SELECT chunk_text, chunk_embedding FROM chunks WHERE file_id = %s", (file_id,))
            for chunk in chunks:
                chunk_id = len(chunk_id_to_text)
                chunk_id_to_text[chunk_id] = chunk['chunk_text']

        # Load Annoy index from database
        annoy_index = None
        dimension = None
        for file_id in file_ids:
            index_blob = execute_query("SELECT index_blob FROM annoy_indexes WHERE file_id = %s", (file_id,))
            if index_blob:
                # Save blob to a temporary file
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(index_blob[0]['index_blob'])
                    temp_file_path = temp_file.name

                # Load Annoy index from the temporary file
                if dimension is None:
                    # Assuming all chunks have the same dimension, fetch the dimension from one of the chunks
                    chunk = execute_query("SELECT chunk_embedding FROM chunks WHERE file_id = %s LIMIT 1", (file_id,))
                    if chunk:
                        dimension = len(json.loads(chunk[0]['chunk_embedding']))
                annoy_index = annoy.AnnoyIndex(dimension, 'angular')
                annoy_index.load(temp_file_path)
                break

        if annoy_index is None or dimension is None:
            return jsonify({"message": "Annoy index not found for the selected files."}), 500

        question_embedding = get_azure_openai_embedding(question)
        question_embedding_np = np.array(question_embedding).astype('float32')  # Ensure it is a 1D array

        nearest_neighbors = annoy_index.get_nns_by_vector(question_embedding_np, 5, include_distances=True)

        relevant_chunks = [chunk_id_to_text[idx] for idx in nearest_neighbors[0]]

        context = "\n\n".join(relevant_chunks)

        return call_openai_api(f"Context:\n{context}\n\nQuestion: {question}")
    except Exception as e:
        error_message = f"Internal server error: {str(e)}"
        print(error_message)  # Log error to console for debugging
        return jsonify({"message": error_message}), 500

@app.route('/summarize', methods=['POST'])
def summarize():
    """
    Route to summarize the content of selected files using Azure OpenAI API.
    """
    project_name = request.form['project_name']
    selected_files = request.form.getlist('files')

    project_id = execute_query("SELECT id FROM projects WHERE name = %s", (project_name,))
    if not project_id:
        return jsonify({"message": f"Project '{project_name}' does not exist."}), 404

    if not selected_files:
        return jsonify({"message": "No files selected."}), 400

    # Fetch the text of the selected files
    file_texts = execute_query(
        "SELECT text FROM files WHERE project_id = %s AND filename IN %s",
        (project_id[0]['id'], selected_files)
    )

    if not file_texts:
        return jsonify({"message": "No text found for the selected files."}), 404

    context = "\n\n".join(file['text'] for file in file_texts)

    try:
        response = call_openai_api(f"Context:\n{context}\n\nPlease provide a summary of the above content.")
        data = response.get_json()  # Get JSON data from the response object
        summary = data.get('answer', "No summary available.")  # Extract the summary from the JSON data
        return jsonify({"summary": summary, "message": "Summary generated successfully"})
    except Exception as e:
        error_message = f"Failed to generate response: {str(e)}"
        print(error_message)  # Log error to console for debugging
        return jsonify({"message": error_message}), 500

@app.route('/get_files', methods=['GET'])
def get_files():
    """
    Route to get the list of files for a given project.
    """
    project_name = request.args.get('project_name')

    project_id = execute_query("SELECT id FROM projects WHERE name = %s", (project_name,))
    if not project_id:
        return jsonify({"message": f"Project '{project_name}' does not exist."}), 404

    files = execute_query("SELECT filename FROM files WHERE project_id = %s", (project_id[0]['id'],))
    file_list = [file['filename'] for file in files]

    return jsonify({"files": file_list}), 200

@app.route('/delete_files', methods=['POST'])
def delete_files():
    """
    Route to delete selected files from a project.
    """
    project_name = request.form['project_name']
    selected_files = request.form.getlist('files')

    project_id = execute_query("SELECT id FROM projects WHERE name = %s", (project_name,))
    if not project_id:
        return jsonify({"message": f"Project '{project_name}' does not exist."}), 404

    try:
        for file in selected_files:
            file_id = execute_query("SELECT id FROM files WHERE project_id = %s AND filename = %s", (project_id[0]['id'], file))
            if file_id:
                # First, delete the associated chunks
                execute_query("DELETE FROM chunks WHERE file_id = %s", (file_id[0]['id'],))
                # Then, delete the file
                execute_query("DELETE FROM files WHERE id = %s", (file_id[0]['id'],))

        return jsonify({"message": "Files deleted successfully"}), 200
    except Exception as e:
        error_message = f"Failed to delete files: {str(e)}"
        print(error_message)  # Log error to console for debugging
        return jsonify({"message": error_message}), 500

def call_openai_api(prompt):
    """
    Helper function to call Azure OpenAI API with a given prompt.
    """
    try:
        completion = client.chat.completions.create(
            model="doc-bot-gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        )

        answer = completion.choices[0].message.content.strip()
        usage_info = completion.model_dump().get('usage')
        completion_json = completion.model_dump_json(indent=2)

        return jsonify({'answer': answer, 'usage': usage_info, 'completion_json': completion_json})
    except Exception as e:
        error_message = f"Failed to generate response: {str(e)}"
        print(error_message)  # Log error to console for debugging
        return jsonify({"message": error_message}), 500

def get_azure_openai_embedding(text):
    """
    Helper function to get embeddings from Azure OpenAI API.
    """
    response = client.embeddings.create(
        input=[text],
        model="doc-bot-embedding-ada-002"
    )
    embedding = response.data[0].embedding
    return embedding

def extract_text(file_stream, file_type):
    """
    Helper function to extract text from a file stream based on file type.
    """
    text = ""
    try:
        if file_type == 'pdf':
            text = ocr_from_pdf(file_stream)  # Use OCR for PDF extraction
        elif file_type == 'html':
            soup = BeautifulSoup(file_stream, 'html.parser')
            text = soup.get_text(separator="\n")
        elif file_type == 'csv':
            file_stream.seek(0)  # Reset stream position
            detected_encoding = chardet.detect(file_stream.read())['encoding']
            file_stream.seek(0)  # Reset stream position again after reading for detection
            # Use a fallback encoding if detection fails
            encoding = detected_encoding if detected_encoding else 'ISO-8859-1'
            text = extract_text_from_csv(file_stream, encoding)
        else:
            raise ValueError("Unsupported file type.")
    except Exception as e:
        print(f"Error extracting text: {e}")
    text = re.sub(r'[^\w\s\.,?]', '', text)
    return text

def ocr_from_pdf(file_stream):
    """
    Helper function to perform OCR on PDF files.
    """
    text = ""
    try:
        file_stream.seek(0)  # Ensure the file stream is at the beginning
        images = convert_from_bytes(file_stream.read())
        for image in images:
            text += pytesseract.image_to_string(image)
    except Exception as e:
        print(f"Error performing OCR on PDF: {e}")
    return text

def extract_text_from_csv(file_stream, encoding):
    """
    Helper function to extract text from CSV files.
    """
    text = ""
    file_stream.seek(0)  # Reset stream position
    content = file_stream.read().decode(encoding).replace('\x00', '')  # Remove NUL bytes
    file_stream = BytesIO(content.encode(encoding))  # Re-create BytesIO stream without NUL bytes
    reader = csv.reader(file_stream.read().decode(encoding).splitlines())
    for row in reader:
        text += ', '.join(row) + '\n'
    return text

def execute_query(query, params=None):
    """
    Helper function to execute a SQL query.
    """
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            result = cursor.fetchall()
        connection.commit()
    finally:
        connection.close()
    return result

def execute_insert(query, params):
    """
    Helper function to execute a SQL insert.
    """
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            connection.commit()
            return cursor.lastrowid
    finally:
        connection.close()

def get_db_connection():
    """
    Helper function to get a database connection.
    """
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='',  # Enter password
        database='docs_chat_bot', # Enter database name
        cursorclass=pymysql.cursors.DictCursor
    )
    return connection

if __name__ == '__main__':
    app.run(debug=True)
