import streamlit as st
import os
from typing import List
import tempfile

from dotenv import load_dotenv

from VectorStore import VectorStore


def save_uploaded_file(uploaded_file) -> str:

    temp_dir = tempfile.gettempdir()
    if uploaded_file is not None:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    return None

def initialize_chat_history() -> List[dict]:
    """Initialize chat history in session state if it doesn't exist"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    return st.session_state.messages

def check_session_state():
    if "COHERE_API_KEY" not in os.environ:
        st.error("Please set the COHERE_API_KEY environment variable")
        return
    if "OPENAI_API_KEY" not in os.environ:
        st.error("Please set the OPENAI_API_KEY environment variable")
        return


def main():
    st.title("RAG Chat Application")
    load_dotenv()
    check_session_state()
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore()

    # Sidebar for PDF upload
    with st.sidebar:
        st.header("Document Upload")
        uploaded_file = st.file_uploader("Upload your PDF document", type=['pdf'])

        if uploaded_file is not None:
            if st.button("Process Document"):
                with st.spinner("Processing document..."):

                    file_path = save_uploaded_file(uploaded_file)

                    # Here you would call your RAG pipeline
                    st.session_state.vector_store.process_document(file_path)

                    st.success("Document processed successfully!")

                    # Clear chat history when new document is processed
                    if "messages" in st.session_state:
                        st.session_state.messages = []

    # Initialize chat history
    messages = initialize_chat_history()

    # Display chat messages
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your document"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Here you would call your RAG query function
                response = st.session_state.vector_store.process_msg(prompt)
                # response = rag_query(prompt)

                st.markdown(response)

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()