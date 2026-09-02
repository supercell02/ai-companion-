import aiosqlite
import json
from datetime import datetime
from typing import Optional, List

class MemoryDB:
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
    
    async def initialize(self):
        """Create tables if they don't exist"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT UNIQUE NOT NULL,
                    category TEXT NOT NULL,
                    embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS personality_traits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trait_name TEXT UNIQUE NOT NULL,
                    trait_value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index on category for faster lookups
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_facts_category 
                ON facts(category)
            """)
            
            await db.commit()
    
    async def store_fact(self, content: str, embedding: list, category: str):
        """Store a fact - with deduplication"""
        embedding_blob = json.dumps(embedding)
        
        async with aiosqlite.connect(self.db_path) as db:
            try:
                # Check if fact already exists
                cursor = await db.execute(
                    "SELECT id FROM facts WHERE content = ?",
                    (content,)
                )
                existing = await cursor.fetchone()
                
                if existing:
                    # Update existing fact
                    await db.execute(
                        """UPDATE facts 
                           SET embedding = ?, category = ?, updated_at = CURRENT_TIMESTAMP,
                               access_count = access_count + 1
                           WHERE id = ?""",
                        (embedding_blob, category, existing[0])
                    )
                    print(f"[UPDATE] Fact already exists: '{content}'")
                else:
                    # Insert new fact
                    await db.execute(
                        """INSERT INTO facts (content, embedding, category) 
                           VALUES (?, ?, ?)""",
                        (content, embedding_blob, category)
                    )
                    print(f"[INSERT] New fact: '{content}'")
                
                await db.commit()
            except Exception as e:
                print(f"Error storing fact: {e}")
                await db.rollback()
    
    async def delete_fact(self, fact_id: int):
        """Delete a fact by ID"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
            await db.commit()
    
    async def delete_fact_by_content(self, content: str):
        """Delete a fact by its content"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM facts WHERE content = ?", (content,))
            await db.commit()
    
    async def get_all_facts(self) -> List[dict]:
        """Retrieve all facts ordered by recency and access"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT id, content, category, created_at, updated_at, access_count
                FROM facts
                ORDER BY updated_at DESC, access_count DESC
            """)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
    
    async def get_similar_facts(self, embedding: list, threshold: float = 0.7, limit: int = 10) -> List[dict]:
        """
        Retrieve facts similar to the query embedding.
        IMPORTANT: Also includes recent facts even if similarity is moderate.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            # Get all facts with embeddings
            cursor = await db.execute("""
                SELECT id, content, category, embedding, created_at, updated_at, access_count
                FROM facts
                ORDER BY updated_at DESC
            """)
            rows = await cursor.fetchall()
            
            scored_facts = []
            
            for row in rows:
                fact_dict = dict(row)
                
                if row['embedding']:
                    embedding_list = json.loads(row['embedding'])
                    similarity = self._cosine_similarity(embedding, embedding_list)
                else:
                    similarity = 0.0
                
                # HYBRID SCORING: Combine similarity + recency + access frequency
                # Recent facts get a boost even if similarity is moderate
                recency_score = self._get_recency_score(row['updated_at'])
                access_score = min(row['access_count'] / 10.0, 1.0)  # Cap at 1.0
                
                # Weighted combination: 50% similarity, 30% recency, 20% access
                combined_score = (similarity * 0.5) + (recency_score * 0.3) + (access_score * 0.2)
                
                fact_dict['similarity'] = similarity
                fact_dict['combined_score'] = combined_score
                scored_facts.append(fact_dict)
            
            # Sort by combined score and return top N
            scored_facts.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # ALWAYS include facts above threshold OR very recent facts
            filtered = [
                f for f in scored_facts 
                if f['similarity'] >= threshold or f['combined_score'] >= 0.6
            ]
            
            # If we have results, return top limit. Otherwise return all top facts.
            return filtered[:limit] if filtered else scored_facts[:limit]
    
    async def get_facts_by_category(self, category: str) -> List[dict]:
        """Get all facts in a specific category"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT id, content, category, created_at, updated_at, access_count
                FROM facts
                WHERE category = ?
                ORDER BY updated_at DESC
            """, (category,))
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
    
    async def store_personality_trait(self, trait_name: str, trait_value: str):
        """Store or update a personality trait"""
        async with aiosqlite.connect(self.db_path) as db:
            try:
                cursor = await db.execute(
                    "SELECT id FROM personality_traits WHERE trait_name = ?",
                    (trait_name,)
                )
                existing = await cursor.fetchone()
                
                if existing:
                    await db.execute(
                        """UPDATE personality_traits 
                           SET trait_value = ?, updated_at = CURRENT_TIMESTAMP
                           WHERE trait_name = ?""",
                        (trait_value, trait_name)
                    )
                else:
                    await db.execute(
                        """INSERT INTO personality_traits (trait_name, trait_value)
                           VALUES (?, ?)""",
                        (trait_name, trait_value)
                    )
                
                await db.commit()
            except Exception as e:
                print(f"Error storing personality trait: {e}")
                await db.rollback()
    
    async def get_personality_traits(self) -> List[dict]:
        """Get all personality traits"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT trait_name, trait_value, updated_at
                FROM personality_traits
                ORDER BY updated_at DESC
            """)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
    
    @staticmethod
    def _cosine_similarity(vec1: list, vec2: list) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a ** 2 for a in vec1) ** 0.5
        norm2 = sum(b ** 2 for b in vec2) ** 0.5
        
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    @staticmethod
    def _get_recency_score(timestamp_str: str) -> float:
        """Convert timestamp to recency score (0-1)"""
        try:
            dt = datetime.fromisoformat(timestamp_str)
            now = datetime.now()
            age_seconds = (now - dt).total_seconds()
            
            # Decay: recent = 1.0, 1 day old = 0.5, 7 days old = 0.1
            max_age = 7 * 24 * 3600  # 7 days in seconds
            recency = max(0.0, 1.0 - (age_seconds / max_age))
            return recency
        except:
            return 0.5  # Default mid-score if parsing fails