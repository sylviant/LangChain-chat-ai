from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

def create_chat():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tu es un assistant IA utile et bienveillant. Réponds toujours en français."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    history = ChatMessageHistory()
    parser = StrOutputParser()

    chain = (
            RunnablePassthrough.assign(
                history=lambda _: history.messages
            )
            | prompt
            | llm
            | parser
    )

    return chain, history


def chat_loop():
    chain, history = create_chat()
    print("=" * 40)
    print("   Chat AI avec Groq + LangChain")
    print("=" * 40)
    print("Tape 'exit' pour quitter.\n")

    while True:
        user_input = input("Toi : ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("À bientôt !")
            break

        print("\nAI : ", end="", flush=True)

        response = ""
        for chunk in chain.stream({"input": user_input}):
            print(chunk, end="", flush=True)
            response += chunk

        print("\n")

        history.add_user_message(user_input)
        history.add_ai_message(response)