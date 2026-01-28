# RAG Architecture
# Docs -> Chunks(splitter)-> VectorDB -> Retrieval
# -> Prompt -> LLM
import os
from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from collections import defaultdict

# LangChain Imports
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ---------------- CONSTANTS ----------------
# We use smaller chunks with more overlap so dates/facts aren't cut in half
CHUNK_SIZE = 600
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "Alibaba-NLP/gte-base-en-v1.5"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

# Global variables to store our "heavy" objects so they don't reload every time
llm = None
vector_store = None

# ---------------- PROMPTS ----------------

# NEW: This prompt forces the AI to explain WHY it doesn't know (e.g., date mismatch)
PROMPT = PromptTemplate(
    template="""
You are a precise Financial Assistant. Answer the question using ONLY the context provided.

Guidelines:
1. If the answer is in the context, state it clearly with the specific date mentioned.
2. If the user asks about the future (like 2026 or 2027) and the context only has forecasts or projections, use those.
3. If the information is truly missing, say: "I'm sorry, the provided documents (covering up to [Latest Date in Context]) do not contain specific information regarding [Topic]." 
4. Do NOT use your own training data or outside knowledge.

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"],
)

CHUNK_SUMMARY_PROMPT = PromptTemplate(
    template="""
Summarize the following text in 1–2 concise sentences focusing only on factual information.
Text: {text}
Summary:
""",
    input_variables=["text"],
)


# ---------------- INITIALIZATION ----------------

def initialize_components():
    """Sets up the AI model and the Database. Runs only once."""
    global llm, vector_store

    if llm is None:
        # llama-3.3-70b is great at following complex instructions
        llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.1,  # Low temperature = less 'creativity', more 'accuracy'
            max_tokens=500
        )

    if vector_store is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(VECTORSTORE_DIR)
        )


# ---------------- INGESTION ----------------

def process_urls(urls):
    """Scrapes websites, splits them into chunks, and saves to Vector DB."""
    initialize_components()

    # Reset DB to ensure we only have fresh data for the current query
    vector_store.reset_collection()

    loader = WebBaseLoader(urls, header_template={"User-Agent": "Mozilla/5.0"})
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )

    docs = splitter.split_documents(data)
    ids = [str(uuid4()) for _ in docs]
    vector_store.add_documents(docs, ids=ids)
    return docs


# ---------------- LOGIC STEPS ----------------

def summarize_chunks(docs):
    """Creates a short summary for each piece of text retrieved."""
    summary_chain = CHUNK_SUMMARY_PROMPT | llm | StrOutputParser()
    return [summary_chain.invoke({"text": doc.page_content}) for doc in docs]


# ---------------- ANSWER GENERATION ----------------

def generate_answer(query: str):
    """
    Retrieves info and generates the final answer.
    Includes 'Query Expansion' to fix the 'I don't know' problem.

    The code in this function:
    Even though we use 3 different questions, we are not trying to get 3 different answers.
    We are trying to find the best possible context (the most relevant paragraphs) to give
    to the LLM so it can give you one single, perfect answer.

    It is like a Trial in Court:

    The User Query is the crime being investigated.
    The 3 Queries are 3 different detectives looking for evidence.

    Detective 1 looks for "2025 rates."

    Detective 2 looks for "Fed projections."

    Detective 3 looks for "Mortgage forecasts."

    They all bring their "evidence" (text chunks) back to the Judge (the LLM).
    The Judge looks at all the evidence gathered by all three detectives and gives one
    final verdict (your answer).

    The Workflow: From 3 Searches to 1 Answer
    Here is exactly how the code merges everything into one:

    * Broad Search: Each of the 3 queries finds 4 chunks. Now you have 12 chunks in a
    big pile.

    * Deduplication: The code checks if "Chunk A" found by Detective 1 is the same as
    "Chunk A" found by Detective 2. If it is, it throws the duplicate away.

    * Ranking: It sorts the remaining chunks so the most relevant ones are at the top.

    * The "Mega-Context": We take the top 6 unique chunks and glue them together
    into one long string of text.

    Example: "Chunk 1 text... \n\n Chunk 2 text... \n\n Chunk 3 text..."

    The Final Call: We send this one "Mega-Context" and your original question to the LLM.

    Why this stops the "I don't know" problem
    By using 3 searches, you are filling the LLM's "brain" with much more
    relevant information.

    If no expanded_query: You asked for "2025." Detective 1 found nothing.
    The LLM had an empty brain and said, "I don't know."

    Now: Detective 1 found nothing for "2025," but Detective 3 found a paragraph about
    "Future projections for 2026." The LLM now has that info and can say,
    "I don't have 2025, but the 2026 forecast says..."

    In simple terms:
    So, first get relavent chunks for my question and two AI variation questions

    Merge and drop duplicates, get top 6 relevant chunks, then send this into llm with
    our prompt (Chain = {context, question} | prompt | llm) and then invoke
    (meaning do what ever you have to do and give me answer).
    """
    if vector_store is None:
        raise RuntimeError("Vector DB not initialized")

    # STEP 1: Query Expansion
    # If user asks "2025 rates", we also search for "interest rate forecasts"
    expansion_query = f"Provide 2 alternative search queries to find information for: {query}"
    expanded_response = llm.invoke(expansion_query).content
    queries = [query] + expanded_response.split('\n')[:2]

    # STEP 2: Retrieval
    # We search the DB for all 3 queries to get a wider range of context
    all_results = []
    for q in queries:
        all_results.extend(vector_store.similarity_search_with_score(q, k=4))

    # STEP 3: Deduplication (Removing duplicate chunks)
    unique_docs = []
    seen_contents = set()
    for doc, score in sorted(all_results, key=lambda x: x[1]):  # Sort by best match
        if doc.page_content not in seen_contents:
            unique_docs.append(doc)
            seen_contents.add(doc.page_content)

    top_docs = unique_docs[:6]  # Take the top 6 most relevant unique chunks

    if not top_docs:
        return "I don't know", None, [], []

    # STEP 4: Build Context and Run Chain
    context_text = "\n\n".join(doc.page_content for doc in top_docs)

    rag_chain = (
            {"context": lambda _: context_text, "question": RunnablePassthrough()}
            | PROMPT
            | llm
            | StrOutputParser()
    )

    answer = rag_chain.invoke(query)

    # Metadata and summaries for the UI/Terminal
    best_source = top_docs[0].metadata.get("source", "unknown")
    chunk_summaries = summarize_chunks(top_docs)

    return answer, best_source, chunk_summaries, top_docs


# ---------------- MAIN RUN ----------------

if __name__ == "__main__":
    # Example: 2026 Outlook Data
    urls = [
        "https://www.freddiemac.com/pmms",
        "https://www.bankrate.com/mortgages/mortgage-rates/",
        "https://www.forbes.com/advisor/mortgages/mortgage-interest-rates-forecast/"
    ]

    print("--- Processing URLs ---")
    process_urls(urls)

    # Testing a "future" question
    user_query = "What is the 30-year mortgage rate as of January 2026?"

    print(f"\nQUERY: {user_query}")
    answer, source, summaries, chunks = generate_answer(user_query)

    print("\n--- FINAL ANSWER ---")
    print(answer)

    print(f"\nPRIMARY SOURCE: {source}")