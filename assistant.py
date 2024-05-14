from typing import Optional

from phi.assistant import Assistant
from phi.knowledge import AssistantKnowledge
from phi.llm.openai import OpenAIChat
from phi.tools.duckduckgo import DuckDuckGo
from phi.embedder.openai import OpenAIEmbedder
from phi.vectordb.pgvector import PgVector2
from phi.storage.assistant.postgres import PgAssistantStorage

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

def get_auto_rag_assistant(
    llm_model: str = "gpt-4-turbo",
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    debug_mode: bool = True
) -> Assistant:
    """Get an Auto RAG Assistant"""

    return Assistant(
        name="auto_rag_assistant",
        run_id=run_id,
        user_id=user_id,
        llm=OpenAIChat(model=llm_model),
        storage=PgAssistantStorage(table_name="auto_rag_assistant_openai", db_url=db_url),
        knowledge_base = AssistantKnowledge(
            vector_db = PgVector2(
                db_url = db_url,
                collection = "auto_rag_document_openai",
                embedder = OpenAIEmbedder(model="text-embedding-3-small", dimensions=1536),

            ),
            num_documents = 3
        ),
        description="You are a helpful Assistant called 'AutoRag' and your goal is to assist the user in best possible way.",
        instructions = [
            "Given a user query, first ALWAYS search your knowledge base using the 'search_knowledge_base' tool to see if you have relevant information.",
            "If you don't find relevant information in your knowledge base, use the 'duckduckgo_search' tool to search the internet.",
            "If you need to reference the chat history, use the 'get_chat_history' tool.",
            "If the user's question is unclear, ask clarifying questions to get more information.",
            "Carefully read the information you have gathered and provide a clear and concise answer to the user.",
            "Do not use phrases like 'based on my knowledge' or 'depending on the information'",
        ],
        show_tool_calls = True,
        search_knowledge = True,
        read_chat_history = True,
        tools = [DuckDuckGo()],
        markdown = True,
        add_chat_history_to_messages = True,
        add_datetime_to_instructions = True,
        debug_mode = debug_mode

    )