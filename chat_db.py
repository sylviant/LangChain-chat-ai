from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from database import (
    nouvelle_conversation, ajouter_message,
    get_messages, get_conversations, supprimer_conversation,
    renommer_conversation
)

load_dotenv()

def create_chain():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tu es un assistant IA utile et bienveillant. Réponds toujours en français."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    parser = StrOutputParser()
    chain = (
            RunnablePassthrough.assign(history=lambda x: x["history"])
            | prompt
            | llm
            | parser
    )
    return chain

def charger_historique(conversation_id):
    messages = get_messages(conversation_id)
    history = []
    for msg in messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))
    return history

def afficher_conversations():
    convs = get_conversations()
    if not convs:
        print("Aucune conversation sauvegardée.")
        return
    print("\nConversations sauvegardées :")
    print("-" * 40)
    for c in convs:
        print(f"  [{c['id']}] {c['titre']} — {c['date']}")
    print()

def chat_loop():
    chain = create_chain()
    conversation_id = None

    print("=" * 40)
    print("   Chat AI avec base de données SQLite")
    print("=" * 40)
    print("Commandes disponibles :")
    print("  /nouvelle       — démarrer une nouvelle conversation")
    print("  /liste          — voir toutes les conversations")
    print("  /charger [id]   — charger une conversation")
    print("  /supprimer [id] — supprimer une conversation")
    print("  /renommer [id] [titre] — renommer une conversation")
    print("  exit            — quitter\n")

    conversation_id = nouvelle_conversation()
    print(f"Nouvelle conversation créée (ID: {conversation_id})\n")

    while True:
        user_input = input("Toi : ").strip()
        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("À bientôt !")
            break

        elif user_input == "/nouvelle":
            conversation_id = nouvelle_conversation()
            print(f"Nouvelle conversation créée (ID: {conversation_id})\n")
            continue

        elif user_input == "/liste":
            afficher_conversations()
            continue

        elif user_input.startswith("/charger"):
            parts = user_input.split()
            if len(parts) == 2 and parts[1].isdigit():
                conversation_id = int(parts[1])
                history = charger_historique(conversation_id)
                print(f"Conversation {conversation_id} chargée ({len(history)} messages)\n")
            else:
                print("Usage : /charger [id]\n")
            continue

        elif user_input.startswith("/supprimer"):
            parts = user_input.split()
            if len(parts) == 2 and parts[1].isdigit():
                supprimer_conversation(int(parts[1]))
                print(f"Conversation {parts[1]} supprimée.\n")
                if int(parts[1]) == conversation_id:
                    conversation_id = nouvelle_conversation()
                    print(f"Nouvelle conversation créée (ID: {conversation_id})\n")
            else:
                print("Usage : /supprimer [id]\n")
            continue

        elif user_input.startswith("/renommer"):
            parts = user_input.split(maxsplit=2)
            if len(parts) == 3 and parts[1].isdigit():
                renommer_conversation(int(parts[1]), parts[2])
                print(f"Conversation {parts[1]} renommée en '{parts[2]}'.\n")
            else:
                print("Usage : /renommer [id] [nouveau titre]\n")
            continue

        history = charger_historique(conversation_id)

        print("\nAI : ", end="", flush=True)
        response = ""
        for chunk in chain.stream({"input": user_input, "history": history}):
            print(chunk, end="", flush=True)
            response += chunk
        print("\n")

        ajouter_message(conversation_id, "user", user_input)
        ajouter_message(conversation_id, "assistant", response)

if __name__ == "__main__":
    chat_loop()