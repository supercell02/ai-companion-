import json
import aiosqlite
from typing import Optional, List


class MemoryDB:
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path

    async def initialize(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    entity_type TEXT NOT NULL DEFAULT 'person',
                    relationship TEXT,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(name, entity_type)
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_id INTEGER,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL,
                    embedding TEXT,
                    confidence REAL DEFAULT 1.0,
                    importance REAL DEFAULT 0.5,
                    access_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_message TEXT NOT NULL,
                    assistant_response TEXT NOT NULL,
                    turn_number INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS personality_traits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trait_name TEXT UNIQUE NOT NULL,
                    trait_value TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS entity_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_entity_id INTEGER NOT NULL,
                    relationship_type TEXT NOT NULL,
                    target_entity_id INTEGER NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_entity_id) REFERENCES entities(id) ON DELETE CASCADE,
                    FOREIGN KEY (target_entity_id) REFERENCES entities(id) ON DELETE CASCADE,
                    UNIQUE(source_entity_id, relationship_type, target_entity_id)
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS memory_access_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact_id INTEGER NOT NULL,
                    query TEXT,
                    similarity REAL,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (fact_id) REFERENCES facts(id) ON DELETE CASCADE
                )
            """)

            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_entities_name
                ON entities(name)
            """)

            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_facts_entity
                ON facts(entity_id)
            """)

            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_facts_category
                ON facts(category)
            """)

            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_facts_updated_at
                ON facts(updated_at)
            """)

            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_relationship_source
                ON entity_relationships(source_entity_id)
            """)

            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_relationship_target
                ON entity_relationships(target_entity_id)
            """)

            await db.commit()

    async def get_or_create_entity(
        self,
        name: str,
        entity_type: str = "person",
        relationship: Optional[str] = None,
        description: Optional[str] = None
    ) -> dict:
        name = name.strip()
        entity_type = entity_type.strip().lower()

        if not name:
            raise ValueError("Entity name cannot be empty")

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            cursor = await db.execute("""
                SELECT
                    id,
                    name,
                    entity_type,
                    relationship,
                    description,
                    created_at,
                    updated_at
                FROM entities
                WHERE LOWER(name) = LOWER(?)
                AND entity_type = ?
                LIMIT 1
            """, (name, entity_type))

            row = await cursor.fetchone()

            if row:
                await db.execute("""
                    UPDATE entities
                    SET
                        relationship = COALESCE(?, relationship),
                        description = COALESCE(?, description),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (
                    relationship,
                    description,
                    row["id"]
                ))

                await db.commit()

                cursor = await db.execute("""
                    SELECT
                        id,
                        name,
                        entity_type,
                        relationship,
                        description,
                        created_at,
                        updated_at
                    FROM entities
                    WHERE id = ?
                """, (row["id"],))

                updated = await cursor.fetchone()
                return dict(updated)

            cursor = await db.execute("""
                INSERT INTO entities (
                    name,
                    entity_type,
                    relationship,
                    description
                )
                VALUES (?, ?, ?, ?)
            """, (
                name,
                entity_type,
                relationship,
                description
            ))

            entity_id = cursor.lastrowid

            await db.commit()

            cursor = await db.execute("""
                SELECT
                    id,
                    name,
                    entity_type,
                    relationship,
                    description,
                    created_at,
                    updated_at
                FROM entities
                WHERE id = ?
            """, (entity_id,))

            row = await cursor.fetchone()
            return dict(row)

    async def get_entity_by_name(
        self,
        name: str,
        entity_type: Optional[str] = None
    ) -> Optional[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            if entity_type:
                cursor = await db.execute("""
                    SELECT
                        id,
                        name,
                        entity_type,
                        relationship,
                        description,
                        created_at,
                        updated_at
                    FROM entities
                    WHERE LOWER(name) = LOWER(?)
                    AND entity_type = ?
                    ORDER BY updated_at DESC
                    LIMIT 1
                """, (
                    name.strip(),
                    entity_type.strip().lower()
                ))
            else:
                cursor = await db.execute("""
                    SELECT
                        id,
                        name,
                        entity_type,
                        relationship,
                        description,
                        created_at,
                        updated_at
                    FROM entities
                    WHERE LOWER(name) = LOWER(?)
                    ORDER BY updated_at DESC
                    LIMIT 1
                """, (name.strip(),))

            row = await cursor.fetchone()

            return dict(row) if row else None

    async def get_entities_by_name(self, name: str) -> List[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            cursor = await db.execute("""
                SELECT
                    id,
                    name,
                    entity_type,
                    relationship,
                    description,
                    created_at,
                    updated_at
                FROM entities
                WHERE LOWER(name) = LOWER(?)
                ORDER BY updated_at DESC
            """, (name.strip(),))

            rows = await cursor.fetchall()

            return [dict(row) for row in rows]

    async def get_all_entities(self) -> List[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            cursor = await db.execute("""
                SELECT
                    id,
                    name,
                    entity_type,
                    relationship,
                    description,
                    created_at,
                    updated_at
                FROM entities
                ORDER BY updated_at DESC
            """)

            rows = await cursor.fetchall()

            return [dict(row) for row in rows]

    async def store_relationship(
        self,
        source_entity_id: int,
        relationship_type: str,
        target_entity_id: int,
        confidence: float = 1.0
    ):
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT id
                FROM entity_relationships
                WHERE source_entity_id = ?
                AND relationship_type = ?
                AND target_entity_id = ?
            """, (
                source_entity_id,
                relationship_type.strip().lower(),
                target_entity_id
            ))

            existing = await cursor.fetchone()

            if existing:
                await db.execute("""
                    UPDATE entity_relationships
                    SET
                        confidence = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (
                    confidence,
                    existing[0]
                ))
            else:
                await db.execute("""
                    INSERT INTO entity_relationships (
                        source_entity_id,
                        relationship_type,
                        target_entity_id,
                        confidence
                    )
                    VALUES (?, ?, ?, ?)
                """, (
                    source_entity_id,
                    relationship_type.strip().lower(),
                    target_entity_id,
                    confidence
                ))

            await db.commit()

    async def get_relationships(
        self,
        entity_id: int
    ) -> List[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            cursor = await db.execute("""
                SELECT
                    r.id,
                    r.relationship_type,
                    r.confidence,
                    r.created_at,
                    r.updated_at,
                    e.id AS target_entity_id,
                    e.name AS target_name,
                    e.entity_type AS target_entity_type,
                    e.relationship AS target_relationship
                FROM entity_relationships r
                JOIN entities e
                    ON e.id = r.target_entity_id
                WHERE r.source_entity_id = ?
                ORDER BY r.updated_at DESC
            """, (entity_id,))

            rows = await cursor.fetchall()

            return [dict(row) for row in rows]

    async def store_fact(
        self,
        content: str,
        embedding: list,
        category: str,
        entity_id: Optional[int] = None,
        confidence: float = 1.0,
        importance: float = 0.5
    ) -> Optional[dict]:
        content = content.strip()
        category = category.strip().lower()

        if not content:
            return None

        embedding_blob = json.dumps(embedding) if embedding else None

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            if entity_id is not None:
                cursor = await db.execute("""
                    SELECT id
                    FROM facts
                    WHERE entity_id = ?
                    AND content = ?
                    LIMIT 1
                """, (entity_id, content))
            else:
                cursor = await db.execute("""
                    SELECT id
                    FROM facts
                    WHERE entity_id IS NULL
                    AND content = ?
                    LIMIT 1
                """, (content,))

            existing = await cursor.fetchone()

            if existing:
                await db.execute("""
                    UPDATE facts
                    SET
                        embedding = COALESCE(?, embedding),
                        category = ?,
                        confidence = ?,
                        importance = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (
                    embedding_blob,
                    category,
                    confidence,
                    importance,
                    existing["id"]
                ))

                fact_id = existing["id"]
            else:
                cursor = await db.execute("""
                    INSERT INTO facts (
                        entity_id,
                        content,
                        category,
                        embedding,
                        confidence,
                        importance
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    entity_id,
                    content,
                    category,
                    embedding_blob,
                    confidence,
                    importance
                ))

                fact_id = cursor.lastrowid

            await db.commit()

            cursor = await db.execute("""
                SELECT
                    f.id,
                    f.entity_id,
                    f.content,
                    f.category,
                    f.confidence,
                    f.importance,
                    f.access_count,
                    f.created_at,
                    f.updated_at,
                    e.name AS entity_name,
                    e.entity_type,
                    e.relationship
                FROM facts f
                LEFT JOIN entities e
                    ON e.id = f.entity_id
                WHERE f.id = ?
            """, (fact_id,))

            row = await cursor.fetchone()

            return dict(row) if row else None

    async def get_fact_by_id(
        self,
        fact_id: int
    ) -> Optional[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            cursor = await db.execute("""
                SELECT
                    f.id,
                    f.entity_id,
                    f.content,
                    f.category,
                    f.embedding,
                    f.confidence,
                    f.importance,
                    f.access_count,
                    f.created_at,
                    f.updated_at,
                    e.name AS entity_name,
                    e.entity_type,
                    e.relationship
                FROM facts f
                LEFT JOIN entities e
                    ON e.id = f.entity_id
                WHERE f.id = ?
            """, (fact_id,))

            row = await cursor.fetchone()

            if not row:
                return None

            result = dict(row)

            if result.get("embedding"):
                result["embedding"] = json.loads(result["embedding"])

            return result

    async def get_all_facts(self) -> List[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            cursor = await db.execute("""
                SELECT
                    f.id,
                    f.entity_id,
                    f.content,
                    f.category,
                    f.confidence,
                    f.importance,
                    f.access_count,
                    f.created_at,
                    f.updated_at,
                    e.name AS entity_name,
                    e.entity_type,
                    e.relationship
                FROM facts f
                LEFT JOIN entities e
                    ON e.id = f.entity_id
                ORDER BY
                    f.updated_at DESC,
                    f.importance DESC
            """)

            rows = await cursor.fetchall()

            return [dict(row) for row in rows]

    async def get_facts_for_entity(
        self,
        entity_id: int,
        limit: int = 50
    ) -> List[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            cursor = await db.execute("""
                SELECT
                    f.id,
                    f.entity_id,
                    f.content,
                    f.category,
                    f.confidence,
                    f.importance,
                    f.access_count,
                    f.created_at,
                    f.updated_at,
                    e.name AS entity_name,
                    e.entity_type,
                    e.relationship
                FROM facts f
                JOIN entities e
                    ON e.id = f.entity_id
                WHERE f.entity_id = ?
                ORDER BY
                    f.importance DESC,
                    f.updated_at DESC
                LIMIT ?
            """, (
                entity_id,
                limit
            ))

            rows = await cursor.fetchall()

            return [dict(row) for row in rows]

    async def get_facts_by_category(
        self,
        category: str
    ) -> List[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            cursor = await db.execute("""
                SELECT
                    f.id,
                    f.entity_id,
                    f.content,
                    f.category,
                    f.confidence,
                    f.importance,
                    f.access_count,
                    f.created_at,
                    f.updated_at,
                    e.name AS entity_name,
                    e.entity_type,
                    e.relationship
                FROM facts f
                LEFT JOIN entities e
                    ON e.id = f.entity_id
                WHERE f.category = ?
                ORDER BY f.updated_at DESC
            """, (category.strip().lower(),))

            rows = await cursor.fetchall()

            return [dict(row) for row in rows]

    async def get_facts_for_entity_and_category(
        self,
        entity_id: int,
        category: str
    ) -> List[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            cursor = await db.execute("""
                SELECT
                    f.id,
                    f.entity_id,
                    f.content,
                    f.category,
                    f.confidence,
                    f.importance,
                    f.access_count,
                    f.created_at,
                    f.updated_at,
                    e.name AS entity_name,
                    e.entity_type,
                    e.relationship
                FROM facts f
                LEFT JOIN entities e
                    ON e.id = f.entity_id
                WHERE f.entity_id = ?
                AND f.category = ?
                ORDER BY f.updated_at DESC
            """, (
                entity_id,
                category.strip().lower()
            ))

            rows = await cursor.fetchall()

            return [dict(row) for row in rows]

    async def get_similar_facts(
        self,
        embedding: list,
        threshold: float = 0.70,
        limit: int = 10,
        entity_id: Optional[int] = None
    ) -> List[dict]:
        if not embedding:
            return []

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            if entity_id is not None:
                cursor = await db.execute("""
                    SELECT
                        f.id,
                        f.entity_id,
                        f.content,
                        f.category,
                        f.embedding,
                        f.confidence,
                        f.importance,
                        f.access_count,
                        f.created_at,
                        f.updated_at,
                        e.name AS entity_name,
                        e.entity_type,
                        e.relationship
                    FROM facts f
                    LEFT JOIN entities e
                        ON e.id = f.entity_id
                    WHERE f.embedding IS NOT NULL
                    AND f.entity_id = ?
                """, (entity_id,))
            else:
                cursor = await db.execute("""
                    SELECT
                        f.id,
                        f.entity_id,
                        f.content,
                        f.category,
                        f.embedding,
                        f.confidence,
                        f.importance,
                        f.access_count,
                        f.created_at,
                        f.updated_at,
                        e.name AS entity_name,
                        e.entity_type,
                        e.relationship
                    FROM facts f
                    LEFT JOIN entities e
                        ON e.id = f.entity_id
                    WHERE f.embedding IS NOT NULL
                """)

            rows = await cursor.fetchall()

        scored = []

        for row in rows:
            fact = dict(row)

            try:
                stored_embedding = json.loads(fact["embedding"])
                similarity = self._cosine_similarity(
                    embedding,
                    stored_embedding
                )
            except Exception:
                similarity = 0.0

            fact.pop("embedding", None)

            fact["similarity"] = similarity

            if similarity >= threshold:
                scored.append(fact)

        scored.sort(
            key=lambda x: x["similarity"],
            reverse=True
        )

        results = scored[:limit]

        if results:
            await self._record_memory_accesses(
                results,
                None
            )

        return results

    async def search_entity_facts(
        self,
        entity_name: str,
        embedding: Optional[list] = None,
        threshold: float = 0.60,
        limit: int = 10
    ) -> List[dict]:
        entity = await self.get_entity_by_name(entity_name)

        if not entity:
            return []

        if embedding:
            return await self.get_similar_facts(
                embedding=embedding,
                threshold=threshold,
                limit=limit,
                entity_id=entity["id"]
            )

        return await self.get_facts_for_entity(
            entity["id"],
            limit
        )

    async def increment_access_count(
        self,
        fact_id: int
    ):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE facts
                SET access_count = access_count + 1
                WHERE id = ?
            """, (fact_id,))

            await db.commit()

    async def _record_memory_accesses(
        self,
        facts: List[dict],
        query: Optional[str]
    ):
        async with aiosqlite.connect(self.db_path) as db:
            for fact in facts:
                fact_id = fact.get("id")

                if not fact_id:
                    continue

                similarity = fact.get("similarity")

                await db.execute("""
                    UPDATE facts
                    SET access_count = access_count + 1
                    WHERE id = ?
                """, (fact_id,))

                await db.execute("""
                    INSERT INTO memory_access_log (
                        fact_id,
                        query,
                        similarity
                    )
                    VALUES (?, ?, ?)
                """, (
                    fact_id,
                    query,
                    similarity
                ))

            await db.commit()

    async def delete_fact(
        self,
        fact_id: int
    ):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                DELETE FROM facts
                WHERE id = ?
            """, (fact_id,))

            await db.commit()

    async def delete_fact_by_content(
        self,
        content: str
    ):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                DELETE FROM facts
                WHERE content = ?
            """, (content,))

            await db.commit()

    async def delete_fact_for_entity(
        self,
        entity_id: int,
        content: str
    ):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                DELETE FROM facts
                WHERE entity_id = ?
                AND content = ?
            """, (
                entity_id,
                content
            ))

            await db.commit()

    async def update_fact(
        self,
        fact_id: int,
        content: Optional[str] = None,
        embedding: Optional[list] = None,
        category: Optional[str] = None,
        confidence: Optional[float] = None,
        importance: Optional[float] = None
    ) -> Optional[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            embedding_blob = (
                json.dumps(embedding)
                if embedding
                else None
            )

            await db.execute("""
                UPDATE facts
                SET
                    content = COALESCE(?, content),
                    embedding = COALESCE(?, embedding),
                    category = COALESCE(?, category),
                    confidence = COALESCE(?, confidence),
                    importance = COALESCE(?, importance),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (
                content,
                embedding_blob,
                category.strip().lower() if category else None,
                confidence,
                importance,
                fact_id
            ))

            await db.commit()

            cursor = await db.execute("""
                SELECT
                    f.id,
                    f.entity_id,
                    f.content,
                    f.category,
                    f.confidence,
                    f.importance,
                    f.access_count,
                    f.created_at,
                    f.updated_at,
                    e.name AS entity_name,
                    e.entity_type,
                    e.relationship
                FROM facts f
                LEFT JOIN entities e
                    ON e.id = f.entity_id
                WHERE f.id = ?
            """, (fact_id,))

            row = await cursor.fetchone()

            return dict(row) if row else None

    async def store_conversation(
        self,
        user_message: str,
        assistant_response: str,
        turn_number: Optional[int] = None
    ):
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                INSERT INTO conversations (
                    user_message,
                    assistant_response,
                    turn_number
                )
                VALUES (?, ?, ?)
            """, (
                user_message,
                assistant_response,
                turn_number
            ))

            conversation_id = cursor.lastrowid

            await db.commit()

            return conversation_id

    async def get_recent_conversations(
        self,
        limit: int = 20
    ) -> List[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            cursor = await db.execute("""
                SELECT
                    id,
                    user_message,
                    assistant_response,
                    turn_number,
                    created_at
                FROM conversations
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))

            rows = await cursor.fetchall()

            results = [dict(row) for row in rows]
            results.reverse()

            return results

    async def store_personality_trait(
        self,
        trait_name: str,
        trait_value: str,
        confidence: float = 1.0
    ):
        trait_name = (
            trait_name
            .strip()
            .lower()
            .replace(" ", "_")
        )

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT id
                FROM personality_traits
                WHERE trait_name = ?
            """, (trait_name,))

            existing = await cursor.fetchone()

            if existing:
                await db.execute("""
                    UPDATE personality_traits
                    SET
                        trait_value = ?,
                        confidence = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (
                    trait_value,
                    confidence,
                    existing[0]
                ))
            else:
                await db.execute("""
                    INSERT INTO personality_traits (
                        trait_name,
                        trait_value,
                        confidence
                    )
                    VALUES (?, ?, ?)
                """, (
                    trait_name,
                    trait_value,
                    confidence
                ))

            await db.commit()

    async def get_personality_traits(self) -> List[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            cursor = await db.execute("""
                SELECT
                    id,
                    trait_name,
                    trait_value,
                    confidence,
                    created_at,
                    updated_at
                FROM personality_traits
                ORDER BY updated_at DESC
            """)

            rows = await cursor.fetchall()

            return [dict(row) for row in rows]

    async def get_memory_stats(self) -> dict:
        async with aiosqlite.connect(self.db_path) as db:
            facts = await db.execute(
                "SELECT COUNT(*) FROM facts"
            )
            facts_count = (await facts.fetchone())[0]

            entities = await db.execute(
                "SELECT COUNT(*) FROM entities"
            )
            entities_count = (await entities.fetchone())[0]

            conversations = await db.execute(
                "SELECT COUNT(*) FROM conversations"
            )
            conversations_count = (await conversations.fetchone())[0]

            traits = await db.execute(
                "SELECT COUNT(*) FROM personality_traits"
            )
            traits_count = (await traits.fetchone())[0]

            relationships = await db.execute(
                "SELECT COUNT(*) FROM entity_relationships"
            )
            relationships_count = (await relationships.fetchone())[0]

        return {
            "facts": facts_count,
            "entities": entities_count,
            "conversations": conversations_count,
            "personality_traits": traits_count,
            "relationships": relationships_count
        }

    async def debug_entity(
        self,
        entity_name: str
    ) -> dict:
        entity = await self.get_entity_by_name(entity_name)

        if not entity:
            return {
                "entity": None,
                "facts": [],
                "relationships": []
            }

        facts = await self.get_facts_for_entity(
            entity["id"]
        )

        relationships = await self.get_relationships(
            entity["id"]
        )

        return {
            "entity": entity,
            "facts": facts,
            "relationships": relationships
        }

    @staticmethod
    def _cosine_similarity(
        vec1: list,
        vec2: list
    ) -> float:
        if not vec1 or not vec2:
            return 0.0

        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(
            a * b
            for a, b in zip(vec1, vec2)
        )

        norm1 = sum(
            a ** 2
            for a in vec1
        ) ** 0.5

        norm2 = sum(
            b ** 2
            for b in vec2
        ) ** 0.5

        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0

        return dot_product / (norm1 * norm2)