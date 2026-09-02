import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import config
from memory.db import MemoryDB
from core.chat_loop import ChatLoop

async def test():
    print("1. Testing database connection...")
    db = MemoryDB(config.database_url)
    
    print("2. Testing fact storage...")
    test_embedding = [0.1] * 768  # Fake embedding for testing
    
    try:
        fact_id = await db.store_fact(
            content="Test fact: user likes pizza",
            embedding=test_embedding,
            category="food"
        )
        print(f"✓ Fact stored with ID: {fact_id}")
    except Exception as e:
        print(f"✗ Failed to store: {e}")
        return
    
    print("3. Testing fact retrieval...")
    try:
        facts = await db.get_all_facts()
        print(f"✓ Retrieved {len(facts)} facts from DB")
        for fact in facts:
            print(f"  - {fact['content']}")
    except Exception as e:
        print(f"✗ Failed to retrieve: {e}")
        return
    
    print("\n4. Testing extraction + embedding...")
    chat_loop = ChatLoop(config)
    
    test_message = "I work as a software engineer and I love Python"
    facts = await chat_loop._extract_facts(test_message)
    print(f"Extracted facts: {facts}")
    
    if facts:
        for fact in facts:
            embedding = await chat_loop._embed_text(fact.get('fact', ''))
            if embedding:
                print(f"✓ Embedded: {fact['fact']}")
            else:
                print(f"✗ Failed to embed: {fact['fact']}")

if __name__ == "__main__":
    asyncio.run(test())