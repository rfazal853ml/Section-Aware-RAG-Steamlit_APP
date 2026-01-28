from langchain_core.prompts import PromptTemplate

score_prompt_template = PromptTemplate(
    input_variables=["query", "summaries_text"],
    template= """You are a precision-oriented information retriever. Your goal is to identify which document contains the specific answer to the User Query.
    
    Scoring Rubric:
    - 90-100: The summary explicitly mentions the core specific entities, dates, or quantitative targets requested in the query.
    - 60-89: The summary covers the exact sub-topic and context but lacks the specific data point or date.
    - 30-59: The summary covers the broad domain but does not address the specific context of the query.
    - 0-29: The summary is unrelated.

    Instructions:
    1. Do not reward "Broad Reviews" if they do not contain the specific details requested.
    2. If a query asks for a "Target" or "Law," look for mentions of "policy," "regulation," or "percentage."
    3. Return ONLY a JSON array: 
    [
    {{"filename": "doc1.pdf", "score": 99}},
    {{"filename": "doc2.pdf", "score": 100}},
    {{"filename": "doc3.pdf", "score": 22}}
    ]
    4. Do NOT include markdown or preamble.

    User Query: 
    {query}

    Summaries:
    {summaries_text}
    """
)

prompt = PromptTemplate(
    input_variables= ["context", "query"],
    template= """You are a research assistant chatbot. Answer the user's question using ONLY the context below.
    
    Context:
    {context}
    
    Question:
    {query}

    Instructions:
    - Provide a clear, explanatory, and accurate answer
    - Only use information from the provided context
    - If the context doesn't contain enough information, say so
    - Be specific and cite details when available

    Answer:"""
) 