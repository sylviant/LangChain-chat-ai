from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()
engine = create_engine("sqlite:///chat_history.db", echo=False)
Session = sessionmaker(bind=engine)

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True)
    titre = Column(String(200))
    created_at = Column(DateTime, default=datetime.now)

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer)
    role = Column(String(20))
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.now)

Base.metadata.create_all(engine)

def nouvelle_conversation(titre="Nouvelle conversation"):
    session = Session()
    conv = Conversation(titre=titre)
    session.add(conv)
    session.commit()
    id_ = conv.id
    session.close()
    return id_

def ajouter_message(conversation_id, role, content):
    session = Session()
    msg = Message(conversation_id=conversation_id, role=role, content=content)
    session.add(msg)
    session.commit()
    session.close()

def get_messages(conversation_id):
    session = Session()
    messages = session.query(Message).filter_by(
        conversation_id=conversation_id
    ).order_by(Message.created_at).all()
    result = [{"role": m.role, "content": m.content} for m in messages]
    session.close()
    return result

def get_conversations():
    session = Session()
    convs = session.query(Conversation).order_by(
        Conversation.created_at.desc()
    ).all()
    result = [{"id": c.id, "titre": c.titre, "date": c.created_at.strftime("%Y-%m-%d %H:%M")} for c in convs]
    session.close()
    return result

def supprimer_conversation(conversation_id):
    session = Session()
    session.query(Message).filter_by(conversation_id=conversation_id).delete()
    session.query(Conversation).filter_by(id=conversation_id).delete()
    session.commit()
    session.close()

def renommer_conversation(conversation_id, nouveau_titre):
    session = Session()
    conv = session.query(Conversation).filter_by(id=conversation_id).first()
    if conv:
        conv.titre = nouveau_titre
        session.commit()
    session.close()

def mettre_a_jour_titre(conversation_id, titre):
    session = Session()
    conv = session.query(Conversation).filter_by(id=conversation_id).first()
    if conv:
        conv.titre = titre
        session.commit()
    session.close()