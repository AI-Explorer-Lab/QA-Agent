import logging
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from db_service.db import query_messages_by_session_id_with_time_order, query_all_messages, init_db
from langchain_core.messages import HumanMessage, AIMessage

init_db()  # 确保表已创建

log = logging.getLogger(__name__)


def load_history_conversation(question: str, session_id: str = None):
    """
    加载历史对话，返回完整的消息列表
    """
    messages = []
    if session_id:
        history_messages = query_messages_by_session_id_with_time_order(session_id)
        for msg in history_messages:
            if msg.role == "user":
                logging.info(f"加载用户消息: {msg.text}")
                messages.append(HumanMessage(content=msg.text))
            elif msg.role == "assistant":
                logging.info(f"加载助手消息: {msg.text}")
                messages.append(AIMessage(content=msg.text))

    # 添加当前用户消息
    messages.append(HumanMessage(content=question))

    logging.debug(f"加载历史对话：{messages}，共 {len(messages)} 条消息")
    return messages


if __name__ == "__main__":
    # 测试历史消息查询
    messages = load_history_conversation("你好", "e9b7c4e5-1348-4bc5-aef8-186791f8319c")
    print(f"加载了 {len(messages)} 条消息:")
    for msg in messages:
        print(f"- {type(msg).__name__}: {msg.content}")
    # query_all_messages()
