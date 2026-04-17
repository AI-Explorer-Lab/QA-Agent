from __future__ import annotations

import json
import os
from datetime import datetime

from core.content_normalizer import normalize_content


def save_conversation_json(session_id, question, answer_text, model_name):
    """
    Save conversation records into a JSON file by session.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    conversations_dir = os.path.join(project_root, "conversations")
    os.makedirs(conversations_dir, exist_ok=True)

    file_path = os.path.join(conversations_dir, f"{session_id}.json")

    new_record = {
        "timestamp": datetime.now().isoformat(),
        "question": normalize_content(question),
        "answer": normalize_content(answer_text),
        "model": model_name,
    }

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            try:
                conversation_data = json.load(file)
            except json.JSONDecodeError:
                conversation_data = []
    else:
        conversation_data = []

    conversation_data.append(new_record)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(conversation_data, file, ensure_ascii=False, indent=2)
