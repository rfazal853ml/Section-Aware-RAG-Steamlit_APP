import json
from typing import List, Tuple, Callable, Optional
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from prompts import  (
    score_prompt_template,
    prompt
)


class RAGSystem:
    def __init__(self, mistral_api_key: str, pinecone_api_key: str, index_name: str):
        """Initialize the RAG system with API keys and index name."""
        self.llm = ChatMistralAI(
            model="mistral-large-latest",
            temperature=0,
            api_key=mistral_api_key
        )
        
        self.embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            api_key=mistral_api_key
        )
        
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(index_name)
        self.vector_store = PineconeVectorStore(
            embedding=self.embeddings,
            index=index
        )
        
    
    def update_status(self, status_callback: Optional[Callable], message: str):
        """Helper to update status if callback provided."""
        if status_callback:
            status_callback.info(f"{message}")
    
    def get_filtered_documents(self, query: str, status_callback: Optional[Callable] = None) -> List[str]:
        """Get filtered document names based on LLM scoring."""
        self.update_status(status_callback, "Retrieving document summaries...")
        
        # Get summary candidates
        summary_candidates = self.vector_store.as_retriever(
            search_kwargs={"k": 1000, "filter": {"type": "summary"}}
        ).invoke(query)
        
        self.update_status(status_callback, "Analyzing documents with AI...")
        
        # Format summaries
        summaries_text = ""
        for doc in summary_candidates:
            summaries_text += f"Filename: {doc.metadata['source']}\nSummary: {doc.page_content}\n---\n"
        
        # Get LLM scores
        response = self.llm.invoke(
            score_prompt_template.format(
                query=query,
                summaries_text=summaries_text
            )
        )
        
        self.update_status(status_callback, "Filtering relevant documents...")
        
        try:
            clean_content = response.content.replace("```json", "").replace("```", "").strip()
            scored_docs = json.loads(clean_content)
            scored_docs.sort(key=lambda x: x["score"], reverse=True)
            
            # Filter docs with score >= 50
            top_scored = [d for d in scored_docs if d["score"] >= 50]
            final_selection = top_scored[:3]
            
            filtered_docs = [d["filename"] for d in final_selection]
            return filtered_docs
        
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return [d.metadata["source"] for d in summary_candidates[:3]]
    
    def retrieve_pages(self, query: str, filtered_docs: List[str], status_callback: Optional[Callable] = None) -> List:
        """Retrieve relevant pages from filtered documents."""
        self.update_status(status_callback, "Retrieving relevant pages...")
        
        pages_retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": 5,
                "filter": {
                    "type": "page",
                    "source": {"$in": filtered_docs}
                }
            }
        )
        return pages_retriever.invoke(query)
    
    def generate_answer(self, query: str, page_hits: List, status_callback: Optional[Callable] = None) -> str:
        """Generate answer from retrieved pages."""
        self.update_status(status_callback, "Generating answer...")
        
        context = "\n\n".join([
            f"[Source: {doc.metadata.get('source', 'Unknown')} | Page {doc.metadata.get('page_number', 0)}]\n{doc.page_content}"
            for doc in page_hits
        ])
        
        
        response = self.llm.invoke(prompt.format(
                query=query,
                context=context
            ))
        return response.content
    
    def extract_sources(self, page_hits: List) -> List[dict]:
        """Extract source metadata from retrieved pages."""
        sources = []
        seen = set()
        
        for doc in page_hits:
            source_key = (doc.metadata.get('source'), doc.metadata.get('page_number'))
            if source_key not in seen:
                sources.append({
                    'source': doc.metadata.get('source', 'Unknown'),
                    'section': doc.metadata.get('section', 'N/A'),
                    'page_number': doc.metadata.get('page_number', 0)
                })
                seen.add(source_key)
        
        return sources
    
    def query(self, user_query: str, status_callback: Optional[Callable] = None) -> Tuple[str, List[dict]]:
        """
        Main query method that orchestrates the RAG pipeline.
        
        Args:
            user_query: The user's question
            status_callback: Optional callback to display status updates
        
        Returns:
            Tuple of (answer, sources)
        """
        try:
            filtered_docs = self.get_filtered_documents(user_query, status_callback)
            
            page_hits = self.retrieve_pages(user_query, filtered_docs, status_callback)
            
            answer = self.generate_answer(user_query, page_hits, status_callback)
            
            sources = self.extract_sources(page_hits)
            
            return answer, sources
        
        except Exception as e:
            return f"An error occurred: {str(e)}", []