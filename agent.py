from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
import math
import datetime

load_dotenv()

search = DuckDuckGoSearchRun()

def recherche_internet(query: str) -> str:
    return search.run(query)

def calculatrice(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, {
            "sqrt": math.sqrt, "pi": math.pi,
            "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "log": math.log,
            "abs": abs, "round": round, "pow": pow,
        })
        return str(result)
    except Exception as e:
        return f"Erreur : {e}"

def date_heure() -> str:
    now = datetime.datetime.now()
    return f"Nous sommes le {now.strftime('%A %d %B %Y')} et il est {now.strftime('%H:%M:%S')}."

def sauvegarde_note(texte: str) -> str:
    with open("notes.txt", "a", encoding="utf-8") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        f.write(f"[{timestamp}] {texte}\n")
    return "Note sauvegardée !"

def executer_outil(nom: str, argument: str) -> str:
    if nom == "recherche_internet":
        print(f"\n Recherche : {argument}")
        result = recherche_internet(argument)
        print(f" Résultat obtenu.")
        return result
    elif nom == "calculatrice":
        print(f"\n Calcul : {argument}")
        result = calculatrice(argument)
        print(f" Résultat : {result}")
        return result
    elif nom == "date_heure":
        print(f"\n Date/heure demandée.")
        return date_heure()
    elif nom == "sauvegarde_note":
        print(f"\n Sauvegarde : {argument}")
        return sauvegarde_note(argument)
    return "Outil inconnu."

def create_agent():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Tu es un assistant IA autonome et intelligent.
Tu as accès aux outils suivants. Quand tu as besoin d'un outil, réponds UNIQUEMENT avec ce format exact :

OUTIL: nom_outil
ARGUMENT: l'argument

Les outils disponibles sont :
- recherche_internet : pour chercher sur le web (argument = la requête)
- calculatrice : pour faire des calculs (argument = l'expression)
- date_heure : pour la date et heure actuelle (argument = rien)
- sauvegarde_note : pour sauvegarder une note (argument = le texte)

Quand tu as la réponse finale, réponds normalement en français sans le format OUTIL/ARGUMENT."""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    chain = prompt | llm | StrOutputParser()
    return chain

def run_agent(user_input, history, chain):
    context = user_input
    max_iterations = 4

    for i in range(max_iterations):
        response = chain.invoke({
            "input": context,
            "history": history.messages
        })

        if "OUTIL:" in response and "ARGUMENT:" in response:
            lines = response.strip().split("\n")
            nom_outil = ""
            argument = ""
            for line in lines:
                if line.startswith("OUTIL:"):
                    nom_outil = line.replace("OUTIL:", "").strip()
                elif line.startswith("ARGUMENT:"):
                    argument = line.replace("ARGUMENT:", "").strip()

            tool_result = executer_outil(nom_outil, argument)

            context = f"""Question originale : {user_input}

Tu as utilisé l'outil '{nom_outil}' et voici le résultat :
{tool_result}

Maintenant réponds à la question en français en utilisant ce résultat."""

        else:
            return response

    return response

def chat_loop():
    chain = create_agent()
    history = ChatMessageHistory()

    print("=" * 40)
    print("   Agent AI autonome avec Groq")
    print("=" * 40)
    print("Outils : recherche web, calculatrice, date/heure, notes")
    print("Tape 'exit' pour quitter.\n")

    while True:
        user_input = input("Toi : ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("À bientôt !")
            break

        print()
        answer = run_agent(user_input, history, chain)
        history.add_user_message(user_input)
        history.add_ai_message(answer)
        print(f"\nAI : {answer}\n")

if __name__ == "__main__":
    chat_loop()