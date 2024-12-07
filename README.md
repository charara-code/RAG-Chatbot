# RAG Chatbot App

This is a Retrieval Augmented Generation (RAG) AI chat app written in streamlit and Langchain.

## Getting started

### Windows

1. **Open a Command Prompt or PowerShell**

    Navigate to your project directory:
   ```cmd
   cd path\to\your\project
   ```

2. **Create a Virtual Environment**

    Run the following command:
    ```cmd
    python -m venv venv
    ```
    This creates a folder named venv in your project directory.

3. **Activate the Virtual Environment**

    Use this command to activate the virtual environment:

    ```cmd
    venv\Scripts\activate
    ```

4. **Verify Activation**

    The terminal prompt will change to indicate that the virtual environment is active (e.g., (venv) at the beginning of the line).


### MacOS and Linux

1. **Open a Terminal**

   Navigate to your project directory:
   ```bash
   cd /path/to/your/project
   ```
2. **Create a Virtual Environment**

    Run the following command:
   ```bash
   python3 -m venv venv
   ```
   This creates a folder named venv in your project directory.

3. **Activate the Virtual Environment**

    Use this command to activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

4. **Verify Activation**

    The terminal prompt will change to indicate that the virtual environment is active (e.g., (venv) at the beginning of the line).

**Note:** to deactivate the virtual environment on Windows, Mac and Linux, simply run:
```bash
deactivate
```


### Installation

To install dependecies, run:
```cmd
pip install -r requirements.txt
```

### Running App

To run the app, run:
```cmd
streamlit run app.py
```

A browser page will open with a UI to upload your PDF documents and chat with.
You can ask questions about your uploaded PDF or just ask any question.
You can add multiple PDF documents and ask questions about any of them (it is currently recommended to use only one PDF per conversation).
   


    