# 📝 **KnowMyRights**

An AI-powered legal assistant that empowers users to instantly access and understand their legal rights, statutes, and case precedents by querying constitutional and statutory documents.

---

## ✨ **Features**

- 📂 **Project Management**: Create and manage multiple projects.
- 📄 **File Uploads**: Upload files like PDFs, HTML, and CSVs.
- 📜 **Text Extraction**: Extract text content from uploaded documents.
- 💬 **Chat with Documents**: Interact with the document content by asking questions.
- 📝 **Summarization**: Generate summaries of selected documents.
- 🗑️ **File Deletion**: Easily remove files from projects.

---

## 🛠️ **Technologies Used**

- **Backend**: Flask, PyMySQL
- **Frontend**: HTML, CSS, Bootstrap, JavaScript
- **Text Processing**: `langchain.text_splitter`, BeautifulSoup, `pytesseract`, `pdf2image`, `chardet`
- **Embedding and Similarity Search**: OpenAI, Annoy
- **Database**: MySQL

---

## ⚙️ **Setup and Installation**

### **Prerequisites**
- 🐍 Python 3.8 or higher
- 🛢️ MySQL server
- 🧠 OpenAI account

### **Installation**

1. **Clone the repository**:
    ```bash
    git clone https://github.com/dhairya1702/FileWhisperer
    cd FileWhisperer
    ```

2. **Set up a virtual environment and activate it**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up the MySQL database**:
    - Create a database named `docs_chat_bot`:
      ```sql
      CREATE DATABASE docs_chat_bot;
      ```
    - Update the MySQL connection details in `app.py`:
      ```python
      connection = pymysql.connect(
          host='localhost',
          user='root',
          password='your_mysql_password',
          database='docs_chat_bot',
          cursorclass=pymysql.cursors.DictCursor
      )
      ```

5. **Configure OpenAI or Azure OpenAI**:
    - Replace the endpoint and key in `app.py` with your Azure OpenAI or OpenAI API details:
      ```python
      azure_openai_endpoint = 'your_azure_openai_endpoint'
      azure_openai_key = 'your_azure_openai_key'
      ```

6. **Run the application**:
    ```bash
    flask run
    ```

---

## 🗂️ **File Structure**

```plaintext
FileWhisperer/
├── app.py              # Main application file
├── templates/          # HTML templates for rendering web pages
├── static/             # Static files (CSS, JavaScript, images)
├── requirements.txt    # List of required Python packages
├── README.md           # Documentation
```

---

## 🚀 Usage

- **Home Page**: View projects and create new ones.
- **Upload Files**: Add files to a project.
- **Chat with Docs**: Interact with content by asking questions.
- **Summarize Files**: Generate summaries of selected files.
- **Delete Files**: Remove files from a project.

---

## 🔗 Routes

| **Route**       | **Description**                                       |
|------------------|-------------------------------------------------------|
| `/`             | Home page displaying projects                         |
| `/add_project`  | Add a new project                                     |
| `/upload`       | Upload a file to a project                            |
| `/chat`         | Chat with documents in a project                      |
| `/summarize`    | Summarize selected files in a project                 |
| `/get_files`    | Get the list of files in a project                    |
| `/delete_files` | Delete selected files from a project                  |

---

## 🤝 Contributing

Contributions are welcome! Please create a pull request with a detailed description of your changes.

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

## 📧 Contact

If you have any questions, feel free to reach out:

- 📩 Email: [dhairya.lalwani2001@gmail.com](mailto:dhairya.lalwani2001@gmail.com)
- 🐙 GitHub: [dhairya1702](https://github.com/dhairya1702)





