import streamlit as st
from rag import graph,  make_vector_store, config
import PyPDF2
from langchain_core.messages import AIMessageChunk

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

with st.sidebar:
    
    pdf_path = st.file_uploader("Upload a PDF file", type=["pdf"], accept_multiple_files=True)
    if pdf_path:
        files_to_add_to_store = []
        for file in pdf_path:
            if file.name not in st.session_state.uploaded_files:
                st.session_state.uploaded_files.append(file.name)
                files_to_add_to_store.append(file)
        if files_to_add_to_store != []:
            with st.spinner("Loading PDFs..."):
                for file in files_to_add_to_store:
                    pdf = PyPDF2.PdfReader(file)
                    pdf_content = ""
                    for page in pdf.pages:
                        pdf_content += page.extract_text()
                    try:
                        make_vector_store({"type": "pdf", "content": pdf_content})
                        st.toast(f"Successfully loaded {file.name}", icon='✌️')
                    except Exception as e:
                        st.toast(f"Failed to load {file.name}", icon="🚨")
            
            files_to_add_to_store = []




            


def response_generator():

    for step in graph.stream(
        {"messages": [{"role": "user", "content": st.session_state.messages[-1]["content"]}]},
        stream_mode="messages", # use "values" to get details of each step
        config=config,
    ): 
        if isinstance(step[0], AIMessageChunk):
            yield step[0].content
        else:
            yield ''


st.title("RAG Chatbot - ESGIA")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])




# Accept user input
if prompt := st.chat_input("Ask me something..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator())
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})