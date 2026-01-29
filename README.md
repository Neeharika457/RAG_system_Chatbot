**🏠 Real Estate AI Sentinel: RAG-Powered Market Research**

**📌 The Problem: "The Information Bottleneck"**
Real estate analysts are drowning in data. To understand market shifts, they must manually scan dozens of articles from sources like CNBC, Realtor.com, and Bloomberg. This process is:

Time-Consuming: Hours spent reading instead of analyzing.

Fragmented: Hard to correlate mortgage trends with local housing inventory.

Risky: High probability of missing "hidden" facts buried in long-form text.

**🚀 The Solution**
The Real Estate AI Sentinel is a Retrieval-Augmented Generation (RAG) tool designed to automate market intelligence. It allows analysts to input URLs from trusted sources and "chat" with the data.

Instead of reading 10 articles, you ask:

"What is the projected impact of current interest rates on first-time homebuyers according to recent reports?"

The system retrieves the specific facts and generates a grounded, cited response in seconds.

**🛠️ How It Works (The RAG Pipeline)**
This project implements a standard RAG architecture to ensure zero hallucinations and high fact-fidelity:

1. Data Ingestion: Scrapes and cleans content from financial and real estate news URLs.

2. Chunking & Embedding: Splits text into semantic segments and converts them into high-dimensional vectors.

3. Vector Store: Stores embeddings in a vector database (e.g., FAISS / Pinecone) for similarity search.

4. Contextual Retrieval: When a query is made, the system finds the most relevant article snippets.

5. Grounded Generation: An LLM synthesizes the answer using only the retrieved context.

**🧩 Tech Stack**
Orchestration: LangChain

Language Model: Groq

Vector Database: Chroma

UI: Streamlit 

**🚦 Getting Started**
1. Clone the repository
Bash
git clone [https://github.com/Neeharika457/RAG_system_Chatbot.git](https://github.com/Neeharika457/RAG_system_Chatbot.git)
cd RAG_System_1

3. Install dependencies
Bash
pip install -r requirements.txt

4. Set up your environment
Create a .env file and add your API keys:
Code snippet
OPENAI_API_KEY=your_key_here

5. Run the application
Bash
streamlit run app.py

**📊 Example Use Cases**
Mortgage Tracking: "Summarize the latest 30-year fixed rate trends from the CNBC articles provided."

Regional Analysis: "Compare housing inventory levels in the Sun Belt versus the Pacific Northwest."

Sentiment Analysis: "What is the general sentiment of experts regarding a potential market correction in 2026?"

**📜 Disclaimer**
This tool is for research purposes only. The summaries generated are based on third-party data and should not be considered financial advice.
