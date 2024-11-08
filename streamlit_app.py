# app.py

import streamlit as st
import logging
from utils import (
    load_documents,
    create_vector_store,
    get_rag_prompt,
    build_rag_chain,
    setup_langgraph,
    invoke_langgraph
)

# Configure logging (if not already configured in utils.py)
logging.basicConfig(
    filename='agentic_rag_app.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(page_title="Agentic Retrieval-Augmented Generation (RAG) App", layout="wide")
    st.title("🧠 Agentic Retrieval-Augmented Generation (RAG) App")
    st.write("""
        This application implements an **Agentic RAG** system using LangChain and LangGraph.
        Enter a query below, and the system will retrieve relevant information from arXiv documents and the web.
    """)

    # User input
    user_query = st.text_input("Enter your query:", "")

    if st.button("Ask"):
        if user_query.strip() == "":
            st.warning("Please enter a valid query.")
            return

        logger.info(f"User Query: {user_query}")

        with st.spinner("Processing your query..."):
            try:
                # Load documents
                documents = load_documents(query=user_query, max_docs=5)

                if not documents:
                    st.error("No documents found for the given query.")
                    logger.warning("No documents found for the given query.")
                    return

                # Create vector store
                vector_store = create_vector_store(documents)

                # Define RAG prompt
                rag_prompt = get_rag_prompt()

                # Build RAG chain
                rag_chain = build_rag_chain(vector_store, rag_prompt)

                # Setup LangGraph
                app = setup_langgraph(vector_store)

                # Invoke LangGraph with user query
                response = invoke_langgraph(app, user_query)

                # Extract and display the response
                messages = response.get('messages', [])
                if messages:
                    final_response = messages[-1].content
                    st.success("🤖 Response:")
                    st.write(final_response)
                    logger.info(f"Response: {final_response}")
                else:
                    st.error("Failed to generate a response.")
                    logger.error("No messages returned in the response.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                logger.error(f"Error during processing: {e}")

    # Display the log file (optional)
    if st.checkbox("Show Logs"):
        st.subheader("Application Logs")
        try:
            with open("agentic_rag_app.log", "r") as log_file:
                logs = log_file.read()
                st.text_area("Logs", logs, height=300)
        except FileNotFoundError:
            st.warning("Log file not found.")

if __name__ == "__main__":
    main()
