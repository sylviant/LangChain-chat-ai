AI Chat with LangChain + Groq
A comprehensive AI chatbot built in Python using LangChain and the Llama 3.3 70B model via Groq (free tier).

Features
Terminal Chat: Real-time streaming interface in the CLI.

Web Interface: Sleek UI powered by Streamlit.

Conversation Memory: Context-aware responses that remember previous interactions.

RAG (Retrieval-Augmented Generation): Chat with your data by asking questions about PDF files.

JSON Persistence: Automatically save and export conversations in JSON format.

git clone https://github.com/your-username/langchain-chat-ai
cd langchain-chat-ai
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

Create a .env file in the root directory

GROQ_API_KEY=gsk_your_api_key_here

Get your free API key at console.groq.com

Usage
Terminal Chat:

Bash
python main.py
Web Interface:

Bash
streamlit run app.py
Chat with a PDF (RAG):

Bash
python rag.py
Chat with History Persistence:

Bash
python chat_with_save.py
Tech Stack
LangChain — AI framework

Groq — Ultra-fast, free LLM API

Llama 3.3 70B — Open-source Large Language Model

Streamlit — Web interface

FAISS — Vector search for RAG
