import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.schemas.chat import ChatRequest, ChatMessage, MessageRole
from app.services.chat_service import chat_service
from app.services.catalog_service import catalog_service

async def run():
    catalog_service.load()
    
    req1 = ChatRequest(messages=[ChatMessage(role=MessageRole.USER, content="I need a Python developer assessment.")])
    res1 = await chat_service.process_message(req1)
    
    print("Reply:", res1.reply[:100] + "...")
    print("End of conversation:", res1.end_of_conversation)
    print("Recommendations:")
    for r in res1.recommendations:
        print(f" - {r.name} ({r.test_type}): {r.url}")

asyncio.run(run())
