# FileWhisperer

FileWhisperer is a web application built using Flask that allows users to upload documents, extract text from them, chat with the document content, and generate summaries. The application leverages OpenAI for text embeddings and chat functionalities, and Spotify's ANNOY for efficient similarity searches.

## Features

- **Project Management**: Create and manage multiple projects.
- **File Uploads**: Upload PDF, HTML, and CSV files to projects.
- **Text Extraction**: Extract text from uploaded documents.
- **Chat with Documents**: Interact with the document content by asking questions.
- **Summarization**: Generate summaries of selected documents.
- **File Deletion**: Remove files from projects.

## Technologies Used

- **Backend**: Flask, PyMySQL
- **Frontend**: HTML, CSS, Bootstrap, JavaScript
- **Text Processing**: langchain.text_splitter, BeautifulSoup, pytesseract, pdf2image, chardet
- **Embedding and Similarity Search**: OpenAI, Annoy
- **Database**: MySQL

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- MySQL server
- OpenAI account

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/dhairya1702/FileWhisperer
    cd FileWhisperer
    ```

2. Set up a virtual environment and activate it:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Set up the MySQL database as done in `DB-Schema.pdf` Create a database named `docs_chat_bot` and configure the connection in the `app.py` file.

    ```sql
    CREATE DATABASE docs_chat_bot;
    ```

    Update the MySQL connection details in `app.py`:

    ```python
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='your_mysql_password',
        database='docs_chat_bot',
        cursorclass=pymysql.cursors.DictCursor
    )
    ```

5. Set up Azure OpenAI. You can also use OpenAI API. Replace the endpoint and key in `app.py` with your Azure OpenAI details:

    ```python
    azure_openai_endpoint = 'your_azure_openai_endpoint'
    azure_openai_key = 'your_azure_openai_key'
    ```

6. Run the application:

    ```bash
    flask run
    ```

### File Structure

- `app.py`: Main application file containing the Flask routes and logic.
- `templates/`: Contains HTML templates for rendering web pages.
- `static/`: Contains static files like CSS and JavaScript.
- `requirements.txt`: List of required Python packages.

## Usage

1. **Home Page**: Displays a list of projects and allows creating new projects.
2. **Upload Files**: Add files to a project by selecting a project and uploading files.
3. **Chat with Docs**: Select files from a project and interact with their content by asking questions.
4. **Summarize Files**: Select files from a project and generate a summary.
5. **Delete Files**: Select files from a project to delete.

### Routes

- `/`: Home page displaying projects.
- `/add_project`: Add a new project.
- `/upload`: Upload a file to a project.
- `/chat`: Chat with documents in a project.
- `/summarize`: Summarize selected files in a project.
- `/get_files`: Get the list of files in a project.
- `/delete_files`: Delete selected files from a project.

## Contributing

Contributions are welcome! Please create a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License. 

## Contact

If you have any questions, feel free to contact me at my email: dhairya.lalwani2001@gmail.com or GitHub: dhairya1702
