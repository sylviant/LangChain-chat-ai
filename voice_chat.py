from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from groq import Groq
from gtts import gTTS
import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import subprocess
import tempfile

load_dotenv()

groq_client = Groq()

def speak(text):
    tts = gTTS(text=text, lang="fr")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        tmp_path = f.name
    tts.save(tmp_path)
    subprocess.run(["start", "/wait", tmp_path], shell=True)
    os.unlink(tmp_path)

def listen():
    print("Parle maintenant (5 secondes)...")
    sample_rate = 16000
    duration = 5
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16"
    )
    sd.wait()
    print("Traitement...")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        tmp_path = f.name
    sf.write(tmp_path, recording, sample_rate)

    with open(tmp_path, "rb") as audio_file:
        transcription = groq_client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_file,
            language="fr"
        )
    os.unlink(tmp_path)

    text = transcription.text.strip()
    if text:
        print(f"Toi : {text}")
        return text
    else:
        print("Aucune voix détectée.")
        return None

def create_chat():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tu es un assistant IA utile et bienveillant. Réponds toujours en français. Garde tes réponses courtes et claires car elles seront lues à voix haute."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    history = ChatMessageHistory()
    parser = StrOutputParser()
    chain = (
            RunnablePassthrough.assign(history=lambda _: history.messages)
            | prompt
            | llm
            | parser
    )
    return chain, history

def chat_loop():
    chain, history = create_chat()
    print("=" * 40)
    print("   Chat vocal AI avec Groq + Whisper")
    print("=" * 40)
    print("Appuie sur ENTRÉE pour parler au micro.")
    print("Tape ton message directement pour le clavier.")
    print("Tape 'exit' pour quitter.\n")

    while True:
        mode = input("[ Entrée = micro | Texte = clavier ] : ").strip()

        if mode.lower() in ("exit", "quit"):
            print("À bientôt !")
            break

        if mode == "":
            user_input = listen()
            if not user_input:
                continue
        else:
            user_input = mode

        print("\nAI : ", end="", flush=True)
        response = ""
        for chunk in chain.stream({"input": user_input}):
            print(chunk, end="", flush=True)
            response += chunk
        print("\n")

        history.add_user_message(user_input)
        history.add_ai_message(response)

        print("Lecture de la réponse...")
        speak(response)

if __name__ == "__main__":
    chat_loop()