import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from config import config
from core.chat_loop import ChatLoop

async def test():
    chat_loop = ChatLoop(config)
    
    test_message = "I'm a software engineer at Google working on the Cloud team. I have a golden retriever named Max. I love Python and competitive programming."
    
    print("Testing fact extraction...")
    print(f"Message: {test_message}\n")
    
    facts = await chat_loop._extract_facts(test_message)
    
    print(f"Extracted facts: {facts}")
    print(f"Number of facts: {len(facts)}")
    
    if facts:
        for i, fact in enumerate(facts, 1):
            print(f"{i}. {fact}")
    else:
        print("NO FACTS EXTRACTED!")

if __name__ == "__main__":
    asyncio.run(test())