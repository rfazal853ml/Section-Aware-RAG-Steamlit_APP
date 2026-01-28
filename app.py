import streamlit as st
from functionality import RAGSystem
import os
from dotenv import load_dotenv

load_dotenv()

# Page config
st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="",
    layout="wide"
)

# Initialize RAG system
@st.cache_resource
def initialize_rag():
    return RAGSystem(
        mistral_api_key=os.environ.get("Mistral_api_key"),
        pinecone_api_key=os.environ.get("Pinecone_api_key"),
        index_name="unstructdocwithmetadata"
    )

rag_system = initialize_rag()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title and clear button
col1, col2 = st.columns([6, 1])
with col1:
    st.title("RAG Research Assistant")
    st.caption("Ask questions about your documents")
with col2:
    st.write("")  # Add spacing
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if available
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            st.markdown("---")
            st.markdown("** Sources:**")
            for i, source in enumerate(message["sources"], 1):
                st.markdown(f"{i}. **{source['source']}** - Section {source['section']} - Page {source['page_number']}")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        response_placeholder = st.empty()
        sources_placeholder = st.empty()
        
        # Query with status updates
        response, sources = rag_system.query(prompt, status_callback=status_placeholder)
        
        # Clear status and show response
        status_placeholder.empty()
        response_placeholder.markdown(response)
        
        # Display sources
        if sources:
            with sources_placeholder.container():
                st.markdown("---")
                st.markdown("** Sources Used: **")
                for i, source in enumerate(sources, 1):
                    st.markdown(f"{i}. **{source['source']}** - **Section:** {source['section']} - **Page:** {source['page_number']}")
    
    # Add assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources
    })