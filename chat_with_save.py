from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import json
import os
from datetime import datetime

load_dotenv()

SAVE_FILE = "conversations.json"

def load_conversations():
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_conversation(messages):
    conversations = load_conversations()
    conversations.append({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "messages": messages
    })
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)
    print(f"\nConversation sauvegardée dans {SAVE_FILE}")

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
    saved_messages = []

    print("=" * 40)
    print("   Chat AI avec sauvegarde JSON")
    print("=" * 40)
    print("Tape 'exit' pour quitter et sauvegarder.\n")

    while True:
        user_input = input("Toi : ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            if saved_messages:
                save_conversation(saved_messages)
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
        saved_messages.append({"role": "user", "content": user_input})
        saved_messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    chat_loop()