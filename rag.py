from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

def load_pdf(path):
    loader = PyPDFLoader(path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(pages)

def create_rag_chain(pdf_path):
    print("Chargement du PDF en cours...")
    docs = load_pdf(pdf_path)

    print("Création des embeddings (peut prendre 1 minute)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Tu es un assistant qui répond aux questions basées uniquement sur le document fourni.
Voici le contexte extrait du document :
{context}

Réponds toujours en français. Si la réponse n'est pas dans le document, dis-le clairement."""),
        ("human", "{question}"),
    ])

    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return chain

if __name__ == "__main__":
    pdf_path = input("Chemin vers ton PDF : ").strip()

    chain = create_rag_chain(pdf_path)
    print("\nPDF chargé avec succès ! Pose tes questions.\n")
    print("=" * 40)

    while True:
        question = input("Toi : ").strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("À bientôt !")
            break

        print("\nAI : ", end="", flush=True)
        for chunk in chain.stream(question):
            print(chunk, end="", flush=True)
        print("\n")