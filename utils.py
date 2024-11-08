# utils.py

import os
import json
import logging
from typing import List, Dict, Any

import arxiv
import faiss
import pymupdf  # For PDF text extraction
from sentence_transformers import SentenceTransformer
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS as LangchainFAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import HumanMessage, AIMessage, FunctionMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain.tools import DuckDuckGoSearchTool, ArxivQueryTool

# Configure logging
logging.basicConfig(
    filename='agentic_rag_app.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)
logger = logging.getLogger(__name__)

def load_documents(query: str, max_docs: int = 5) -> List[Dict[str, Any]]:
    """
    Loads documents from arXiv based on the provided query.

    Args:
        query (str): The search query.
        max_docs (int): Maximum number of documents to retrieve.

    Returns:
        List[Dict[str, Any]]: A list of documents with content and metadata.

    Example:
        >>> documents = load_documents("Retrieval Augmented Generation", max_docs=2)
        >>> len(documents)
        2
        >>> documents[0].keys()
        dict_keys(['title', 'summary', 'authors', 'published', 'content'])
    """
    logger.info(f"Loading documents with query: {query} and max_docs: {max_docs}")
    search = arxiv.Search(
        query=query,
        max_results=max_docs,
        sort_by=arxiv.SortCriterion.Relevance
    )

    documents = []
    for result in search.results():
        try:
            # Download PDF
            pdf_response = arxiv.download_pdf(result)
            pdf_path = f"temp_{result.get_short_id()}.pdf"
            with open(pdf_path, 'wb') as f:
                f.write(pdf_response.read())
            
            # Extract text from PDF
            content = extract_text_from_pdf(pdf_path)
            if not content.strip():
                logger.warning(f"No content extracted from {pdf_path}. Skipping document.")
                os.remove(pdf_path)
                continue

            doc = {
                'title': result.title,
                'summary': result.summary,
                'authors': [author.name for author in result.authors],
                'published': result.published.strftime('%Y-%m-%d'),
                'content': content
            }
            documents.append(doc)
            logger.info(f"Loaded document: {doc['title']}")
            
            # Remove temporary PDF file
            os.remove(pdf_path)
        except Exception as e:
            logger.error(f"Error loading document {result.title}: {e}")
            continue

    return documents

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file using PyMuPDF.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.

    Example:
        >>> text = extract_text_from_pdf("sample.pdf")
        >>> isinstance(text, str)
        True
    """
    logger.info(f"Extracting text from PDF: {pdf_path}")
    try:
        doc = pymupdf.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        logger.info(f"Extracted text from {pdf_path}")
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {e}")
        return ""

def create_vector_store(documents: List[Dict[str, Any]], embedding_model_name: str = 'all-MiniLM-L6-v2') -> LangchainFAISS:
    """
    Creates a FAISS vector store from document embeddings.

    Args:
        documents (List[Dict[str, Any]]): List of documents.
        embedding_model_name (str): Name of the sentence transformer model.

    Returns:
        LangchainFAISS: FAISS vector store containing document embeddings.

    Example:
        >>> vector_store = create_vector_store(documents, "all-MiniLM-L6-v2")
        >>> isinstance(vector_store, LangchainFAISS)
        True
    """
    logger.info("Creating embeddings and building FAISS vector store.")
    # Initialize embedding model
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    # Combine all document contents for embedding
    texts = [doc['content'] for doc in documents]
    
    # Create FAISS index
    vector_store = LangchainFAISS.from_texts(texts, embeddings)
    logger.info("FAISS vector store created successfully.")
    return vector_store

def get_rag_prompt() -> ChatPromptTemplate:
    """
    Defines the RAG prompt template.

    Returns:
        ChatPromptTemplate: The RAG prompt template.

    Example:
        >>> prompt = get_rag_prompt()
        >>> isinstance(prompt, ChatPromptTemplate)
        True
    """
    logger.info("Defining RAG prompt template.")
    RAG_PROMPT = """\
Use the following context to answer the user's query. If you cannot answer the question, please respond with 'I don't know'.

Question:
{question}

Context:
{context}
"""
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    logger.info("RAG prompt template defined.")
    return prompt

def build_rag_chain(vector_store: LangchainFAISS, prompt: ChatPromptTemplate, model_name: str = 'gpt-3.5-turbo') -> LLMChain:
    """
    Builds the Retrieval-Augmented Generation (RAG) chain.

    Args:
        vector_store (LangchainFAISS): The FAISS vector store.
        prompt (ChatPromptTemplate): The RAG prompt.
        model_name (str): The language model to use.

    Returns:
        LLMChain: The RAG chain.

    Example:
        >>> rag_chain = build_rag_chain(vector_store, prompt, "gpt-3.5-turbo")
        >>> isinstance(rag_chain, LLMChain)
        True
    """
    logger.info("Building the RAG chain.")
    # Initialize the language model
    llm = ChatOpenAI(model=model_name, temperature=0)
    
    # Create the chain
    chain = LLMChain(llm=llm, prompt=prompt)
    logger.info("RAG chain built successfully.")
    return chain

def setup_langgraph(vector_store: LangchainFAISS) -> Any:
    """
    Sets up the LangGraph for Agentic RAG.

    Args:
        vector_store (LangchainFAISS): The FAISS vector store.

    Returns:
        Any: The compiled LangGraph application.

    Example:
        >>> app = setup_langgraph(vector_store)
        >>> app
        <CompiledGraph object>
    """
    logger.info("Setting up LangGraph for Agentic RAG.")
    
    # Initialize tools
    tools = [
        DuckDuckGoSearchTool(),
        ArxivQueryTool()
    ]
    tool_executor = ToolExecutor(tools)
    
    # Initialize the language model
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Define functions based on tools
    functions = [tool.to_function() for tool in tools]
    llm.bind_functions(functions)
    
    # Define AgentState
    from typing import TypedDict, Sequence
    from langchain.schema import BaseMessage
    
    class AgentState(TypedDict):
        messages: Sequence[BaseMessage]
    
    # Define node functions
    def call_model(state: AgentState) -> Dict[str, Any]:
        """
        Invokes the language model with the current messages.

        Args:
            state (AgentState): The current state containing messages.

        Returns:
            Dict[str, Any]: Updated state with the AI's response.

        Example:
            >>> new_state = call_model(state)
            >>> 'messages' in new_state
            True
        """
        messages = state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}
    
    def call_tool(state: AgentState) -> Dict[str, Any]:
        """
        Invokes a tool based on the last function call in messages.

        Args:
            state (AgentState): The current state containing messages.

        Returns:
            Dict[str, Any]: Updated state with the tool's response.

        Example:
            >>> new_state = call_tool(state)
            >>> 'messages' in new_state
            True
        """
        last_message = state["messages"][-1]
        if isinstance(last_message, FunctionMessage):
            tool_name = last_message.name
            tool_input = json.loads(last_message.content)
            response = tool_executor.invoke(tool_name, tool_input)
            function_message = FunctionMessage(content=str(response), name=tool_name)
            return {"messages": [function_message]}
        return {}
    
    # Initialize StateGraph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("action", call_tool)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Define conditional edges
    def should_continue(state: AgentState) -> str:
        """
        Determines whether to continue to the action node or end the workflow.

        Args:
            state (AgentState): The current state containing messages.

        Returns:
            str: "continue" or "end".

        Example:
            >>> decision = should_continue(state)
            >>> decision in ["continue", "end"]
            True
        """
        last_message = state["messages"][-1]
        if isinstance(last_message, FunctionMessage):
            return "continue"
        return "end"
    
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "end": END
        }
    )
    
    # Connect action back to agent
    workflow.add_edge("action", "agent")
    
    # Compile the workflow
    app = workflow.compile()
    logger.info("LangGraph setup completed.")
    return app

def invoke_langgraph(app: Any, user_query: str) -> Dict[str, Any]:
    """
    Invokes the LangGraph application with the user's query.

    Args:
        app (Any): The compiled LangGraph application.
        user_query (str): The user's input query.

    Returns:
        Dict[str, Any]: The response from the LangGraph application.

    Example:
        >>> response = invoke_langgraph(app, "What is RAG?")
        >>> 'messages' in response
        True
    """
    logger.info(f"Invoking LangGraph with query: {user_query}")
    inputs = {"messages": [HumanMessage(content=user_query)]}
    response = app.invoke(inputs)
    logger.info("LangGraph invoked successfully.")
    return response
