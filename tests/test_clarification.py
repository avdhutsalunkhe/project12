import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.schemas.chat import ChatRequest, ChatMessage, MessageRole
from app.services.chat_service import chat_service
from app.services.catalog_service import catalog_service
from app.services.retrieval_service import retrieval_service

async def run():
    catalog_service.load()
    
    req1 = ChatRequest(messages=[ChatMessage(role=MessageRole.USER, content="I need an assessment.")])
    res1 = await chat_service.process_message(req1)
    print("Vague request reply:", res1.reply)
    
    req2 = ChatRequest(messages=[
        ChatMessage(role=MessageRole.USER, content="I need an assessment."),
        ChatMessage(role=MessageRole.ASSISTANT, content="What role or position are you looking to assess candidates for?"),
        ChatMessage(role=MessageRole.USER, content="A Python developer.")
    ])
    res2 = await chat_service.process_message(req2)
    print("\nMore specific request reply:", res2.reply)

asyncio.run(run())
