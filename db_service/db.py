# db.py
from __future__ import annotations

import datetime
import os

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from core.config_loader import get_llm_runtime_config
from core.content_normalizer import normalize_content


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_URL = f"sqlite:///{BASE_DIR}/database/chat-llm.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    role = Column(String, default="user")
    text = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


def init_db() -> None:
    db_dir = os.path.join(BASE_DIR, "database")
    os.makedirs(db_dir, exist_ok=True)
    Base.metadata.create_all(bind=engine)


def save_conversation_sql(
    session_id: str,
    question: str,
    answer_text: str,
    model_name: str = "",
) -> bool:
    if not model_name:
        model_name = get_llm_runtime_config().get("model", "")

    db = SessionLocal()
    try:
        question_text = normalize_content(question)
        answer_plain_text = normalize_content(answer_text)

        user_message = Message(
            session_id=session_id,
            role="user",
            text=question_text,
            created_at=datetime.datetime.utcnow(),
        )
        db.add(user_message)

        ai_message = Message(
            session_id=session_id,
            role="assistant",
            text=answer_plain_text,
            created_at=datetime.datetime.utcnow(),
        )
        db.add(ai_message)

        db.commit()
        return True
    except Exception as exc:
        db.rollback()
        print(f"保存对话记录时出错: {exc}")
        return False
    finally:
        db.close()


def query_messages_by_session_id_with_time_order(session_id: str):
    init_db()
    db = SessionLocal()
    try:
        return (
            db.query(Message)
            .filter(Message.session_id == session_id)
            .order_by(Message.created_at.asc())
            .all()
        )
    finally:
        db.close()


def query_all_messages() -> None:
    init_db()
    db = SessionLocal()
    try:
        messages = db.query(Message).all()
        for msg in messages:
            print(f"SessionId: {msg.session_id}")
            print(f"Role: {msg.role}")
            print(f"text: {msg.text}")
            print(f"created_at: {msg.created_at}")
            print("-" * 30)
    finally:
        db.close()


if __name__ == "__main__":
    results = query_messages_by_session_id_with_time_order("dev-test")
    if results:
        print("AI 最新的一条回复是：", results[-1].text)
    else:
        print("没有找到匹配的记录")
