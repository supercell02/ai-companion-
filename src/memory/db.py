import sqlite3
import json
from typing import List, Dict, Optional
import asyncio

class MemoryDB:
    def __init__(self, db_path: str = "companion_memory.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL UNIQUE,
                    category TEXT,
                    embedding TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS personality_traits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trait_name TEXT NOT NULL UNIQUE,
                    trait_value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def _get_connection(self):
        """Get DB connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    async def store_fact(self, content: str, embedding: List[float], category: str):
        """Store or update a fact"""
        def _insert():
            with self._get_connection() as conn:
                try:
                    conn.execute(
                        """INSERT INTO facts (content, embedding, category) 
                           VALUES (?, ?, ?)""",
                        (content, json.dumps(embedding), category)
                    )
                    conn.commit()
                    fact_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                    print(f"[DB] Stored fact: ID={fact_id}, '{content}'")
                    return fact_id
                except sqlite3.IntegrityError:
                    # Fact already exists, update it
                    conn.execute(
                        """UPDATE facts 
                           SET embedding = ?, category = ?, updated_at = CURRENT_TIMESTAMP
                           WHERE content = ?""",
                        (json.dumps(embedding), category, content)
                    )
                    conn.commit()
                    print(f"[DB] Updated fact: '{content}'")
                    return conn.execute("SELECT id FROM facts WHERE content = ?", (content,)).fetchone()[0]
        
        try:
            return await asyncio.to_thread(_insert)
        except Exception as e:
            print(f"[DB Error] {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def get_similar_facts(self, embedding: List[float], threshold: float = 0.7, limit: int = 5) -> List[Dict]:
        """Retrieve facts via embedding similarity"""
        def _search():
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT id, content, category, embedding FROM facts")
                facts = cursor.fetchall()
                
                print(f"[DB DEBUG] Total facts in DB: {len(facts)}")
                
                # Calculate cosine similarity
                from numpy import dot
                from numpy.linalg import norm
                
                user_embedding = embedding
                results = []
                
                for fact in facts:
                    fact_embedding = json.loads(fact['embedding'])
                    
                    # Cosine similarity
                    numerator = dot(user_embedding, fact_embedding)
                    denominator = norm(user_embedding) * norm(fact_embedding)
                    
                    if denominator > 0:
                        similarity = numerator / denominator
                        print(f"[DB DEBUG] Fact: '{fact['content'][:50]}...' - Similarity: {similarity:.3f}")
                        
                        if similarity > threshold:
                            results.append({
                                'id': fact['id'],
                                'content': fact['content'],
                                'category': fact['category'],
                                'similarity': similarity
                            })
                
                print(f"[DB DEBUG] After filtering (threshold={threshold}): {len(results)} facts")
                
                # Sort by similarity
                results.sort(key=lambda x: x['similarity'], reverse=True)
                return results[:limit]
        
        try:
            return await asyncio.to_thread(_search)
        except Exception as e:
            print(f"[DB Error] {e}")
            import traceback
            traceback.print_exc()
            return []
    async def get_all_facts(self) -> List[Dict]:
        """Get all facts"""
        def _fetch():
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT id, content, category, embedding FROM facts")
                return [dict(row) for row in cursor.fetchall()]
        
        try:
            return await asyncio.to_thread(_fetch)
        except Exception as e:
            print(f"[DB Error] {e}")
            return []
    
    async def delete_fact(self, fact_id: int):
        """Delete a fact"""
        def _delete():
            with self._get_connection() as conn:
                conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
                conn.commit()
                print(f"[DB] Deleted fact ID={fact_id}")
        
        try:
            await asyncio.to_thread(_delete)
        except Exception as e:
            print(f"[DB Error] {e}")
    
    async def store_personality_trait(self, trait_name: str, trait_value: str):
        """Store or update personality trait"""
        def _upsert():
            with self._get_connection() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO personality_traits (trait_name, trait_value) 
                       VALUES (?, ?)""",
                    (trait_name, trait_value)
                )
                conn.commit()
                print(f"[DB] Stored trait: '{trait_name}' = '{trait_value}'")
        
        try:
            await asyncio.to_thread(_upsert)
        except Exception as e:
            print(f"[DB Error] {e}")
    
    async def get_personality_traits(self) -> List[Dict]:
        """Get all personality traits"""
        def _fetch():
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT trait_name, trait_value FROM personality_traits")
                return [dict(row) for row in cursor.fetchall()]
        
        try:
            return await asyncio.to_thread(_fetch)
        except Exception as e:
            print(f"[DB Error] {e}")
            return []