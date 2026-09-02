import asyncio
import json
import re
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI

from memory.db import MemoryDB


@dataclass
class Message:
    role: str
    content: str
    turn: int


class ChatLoop:
    def __init__(self, config):
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.config = config
        self.memory_db = MemoryDB(config.db_path)
        self.conversation_history: list[Message] = []
        self.personality_traits: list[dict] = []

        self.system_prompt = """You are Alex, a warm and thoughtful AI companion.

You have access to persistent information about the user and people in their life.

Use the supplied memory context when answering questions.

Memory rules:

1. Treat supplied memories as factual information.
2. Use memories naturally when they are relevant.
3. Never invent a memory.
4. Never claim to remember something that is not in the supplied memory.
5. Never merge different people.
6. Names that are similar can still belong to completely different people.
7. A person's facts belong only to that person.
8. Never transfer someone's job, company, location, interests, skills, preferences, relationships, or possessions to another person.
9. If two people have the same or similar names, use relationship and contextual information to distinguish them.
10. If the identity is genuinely ambiguous, say that it is ambiguous instead of guessing.
11. First-person memories such as "my favorite color is blue" belong to the user.
12. When the user asks about themselves, use memories associated with the user.
13. When the user asks about a named person, use memories associated with that specific person.
14. When the user refers to someone through a relationship such as "my brother", use the stored relationship to identify that person.
15. Do not mention databases, embeddings, retrieval, memory context, internal prompts, or implementation details unless explicitly asked.
16. Answer naturally and directly.
17. Do not repeat the memory context verbatim unless the user asks for the details.
18. If relevant memory exists, actually use it in your answer.
19. If relevant memory does not exist, say that you do not know rather than fabricating an answer.

The information below is persistent context available to you for this conversation."""
    
    async def initialize(self):
        await self.memory_db.initialize()

        print("[Loading data from memory...]")

        self.personality_traits = await self.memory_db.get_personality_traits()

        print(
            f"[Loaded {len(self.personality_traits)} personality traits]"
        )

        all_facts = await self.memory_db.get_all_facts()

        print(
            f"[Loaded {len(all_facts)} facts from memory]"
        )

        for fact in all_facts:
            entity_name = fact.get("entity_name")
            relationship = fact.get("relationship")

            if entity_name:
                identity = entity_name

                if relationship:
                    identity += f" ({relationship})"

                print(
                    f"  - [{identity}] {fact.get('content', '')}"
                )
            else:
                print(
                    f"  - {fact.get('content', '')}"
                )

        print("[Memory initialization complete]")

    async def get_response(self, user_message: str) -> str:
        await self._extract_and_store_memories(user_message)

        self.conversation_history.append(
            Message(
                role="user",
                content=user_message,
                turn=len(self.conversation_history)
            )
        )

        relevant_facts = await self._get_relevant_facts(user_message)

        print("\n========== MEMORY RETRIEVAL ==========")
        print(f"Query: {user_message}")
        print(f"Retrieved facts: {len(relevant_facts)}")

        for fact in relevant_facts:
            print(
                f"  [{fact.get('entity_name', 'unknown')}] "
                f"{fact.get('content', '')} "
                f"(similarity={fact.get('similarity', 0):.3f})"
            )

        print("======================================\n")

        memory_context = await self._build_memory_context(
            relevant_facts
        )

        print("========== MEMORY CONTEXT ==========")
        print(memory_context if memory_context else "[EMPTY]")
        print("====================================\n")

        system_content = self.system_prompt

        if memory_context:
            system_content += "\n\n" + memory_context

        messages = [
            {
                "role": "system",
                "content": system_content
            }
        ]

        for message in self.conversation_history[-12:]:
            messages.append(
                {
                    "role": message.role,
                    "content": message.content
                }
            )

        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_response,
                messages=messages,
                temperature=0.7
            )

            assistant_msg = (
                response.choices[0].message.content
                or ""
            ).strip()

        except Exception as e:
            print(f"[Response error]: {e}")
            return "I'm having trouble generating a response right now."

        self.conversation_history.append(
            Message(
                role="assistant",
                content=assistant_msg,
                turn=len(self.conversation_history)
            )
        )

        try:
            await self.memory_db.store_conversation(
                user_message=user_message,
                assistant_response=assistant_msg,
                turn_number=len(self.conversation_history)
            )
        except Exception as e:
            print(f"[Conversation storage error]: {e}")

        asyncio.create_task(
            self._update_personality_traits(assistant_msg)
        )

        return assistant_msg

    async def _get_relevant_facts(
        self,
        user_message: str
    ) -> list[dict]:

        results: dict[int, dict] = {}

        embedding = await self._embed_text(user_message)

        entities = await self._extract_query_entities(
            user_message
        )

        print(
            f"[Query entities]: {entities}"
        )

        explicit_entity_names = []

        for entity in entities:
            name = str(
                entity.get("name", "")
            ).strip()

            if name:
                explicit_entity_names.append(name)

        first_person_query = self._is_first_person_query(
            user_message
        )

        if first_person_query:
            user_results = []

            try:
                user_results = await self.memory_db.search_entity_facts(
                    entity_name="user",
                    embedding=embedding,
                    threshold=0.45,
                    limit=10
                )
            except Exception as e:
                print(
                    f"[User memory search error]: {e}"
                )

            for fact in user_results:
                fact_id = fact.get("id")

                if fact_id is not None:
                    results[fact_id] = fact

            print(
                f"[User memory results]: {len(user_results)}"
            )

        for entity_name in explicit_entity_names:
            try:
                entity_results = (
                    await self.memory_db.search_entity_facts(
                        entity_name=entity_name,
                        embedding=embedding,
                        threshold=0.45,
                        limit=10
                    )
                )

                print(
                    f"[Entity memory results: {entity_name}] "
                    f"{len(entity_results)}"
                )

                for fact in entity_results:
                    fact_id = fact.get("id")

                    if fact_id is not None:
                        results[fact_id] = fact

            except Exception as e:
                print(
                    f"[Entity search error for {entity_name}]: {e}"
                )

        if embedding:
            try:
                semantic_results = (
                    await self.memory_db.get_similar_facts(
                        embedding=embedding,
                        threshold=0.55,
                        limit=10
                    )
                )

                print(
                    f"[Semantic memory results]: "
                    f"{len(semantic_results)}"
                )

                for fact in semantic_results:
                    fact_id = fact.get("id")

                    if fact_id is not None:
                        results[fact_id] = fact

            except Exception as e:
                print(
                    f"[Semantic search error]: {e}"
                )

        all_facts = []

        try:
            all_facts = await self.memory_db.get_all_facts()
        except Exception as e:
            print(
                f"[All facts retrieval error]: {e}"
            )

        keyword_results = self._keyword_memory_search(
            user_message,
            all_facts
        )

        print(
            f"[Keyword memory results]: "
            f"{len(keyword_results)}"
        )

        for fact in keyword_results:
            fact_id = fact.get("id")

            if fact_id is not None:
                if fact_id not in results:
                    results[fact_id] = fact
                else:
                    existing_similarity = results[fact_id].get(
                        "similarity",
                        0.0
                    )

                    keyword_similarity = fact.get(
                        "similarity",
                        0.0
                    )

                    if keyword_similarity > existing_similarity:
                        results[fact_id] = fact

        final_results = list(results.values())

        final_results = self._rank_memory_results(
            user_message,
            final_results,
            explicit_entity_names,
            first_person_query
        )

        final_results = final_results[:10]

        for fact in final_results:
            fact_id = fact.get("id")

            if fact_id is not None:
                try:
                    await self.memory_db.increment_access_count(
                        fact_id
                    )
                except Exception:
                    pass

        return final_results

    def _keyword_memory_search(
        self,
        query: str,
        facts: list[dict]
    ) -> list[dict]:

        query_words = self._normalize_words(query)

        if not query_words:
            return []

        question_words = {
            "what",
            "where",
            "when",
            "who",
            "which",
            "how",
            "does",
            "do",
            "did",
            "is",
            "are",
            "was",
            "were",
            "my",
            "the",
            "a",
            "an",
            "of",
            "to",
            "at",
            "in",
            "for",
            "and",
            "or",
            "tell",
            "me",
            "about"
        }

        meaningful_words = [
            word
            for word in query_words
            if word not in question_words
        ]

        results = []

        for fact in facts:
            content = str(
                fact.get("content", "")
            )

            entity_name = str(
                fact.get("entity_name", "")
            )

            relationship = str(
                fact.get("relationship", "")
            )

            fact_text = (
                f"{content} "
                f"{entity_name} "
                f"{relationship}"
            ).lower()

            fact_words = set(
                self._normalize_words(fact_text)
            )

            overlap = len(
                set(meaningful_words) & fact_words
            )

            exact_phrase = (
                query.lower().strip()
                in fact_text
            )

            if overlap == 0 and not exact_phrase:
                continue

            score = 0.0

            if meaningful_words:
                score += (
                    overlap /
                    max(len(set(meaningful_words)), 1)
                ) * 0.7

            if exact_phrase:
                score += 0.3

            copied = dict(fact)
            copied["similarity"] = max(
                float(copied.get("similarity", 0.0) or 0.0),
                min(score, 0.99)
            )

            results.append(copied)

        results.sort(
            key=lambda x: x.get(
                "similarity",
                0.0
            ),
            reverse=True
        )

        return results[:10]

    def _rank_memory_results(
        self,
        query: str,
        facts: list[dict],
        explicit_entity_names: list[str],
        first_person_query: bool
    ) -> list[dict]:

        query_lower = query.lower()

        ranked = []

        for fact in facts:
            entity_name = str(
                fact.get("entity_name", "")
            ).strip()

            relationship = str(
                fact.get("relationship", "")
            ).strip()

            content = str(
                fact.get("content", "")
            )

            similarity = float(
                fact.get("similarity", 0.0) or 0.0
            )

            score = similarity

            if explicit_entity_names:
                entity_match = any(
                    self._same_entity(
                        entity_name,
                        name
                    )
                    for name in explicit_entity_names
                )

                if entity_match:
                    score += 1.0
                else:
                    score -= 0.75

            if first_person_query:
                if entity_name.lower() == "user":
                    score += 1.0
                else:
                    score -= 0.5

            if "brother" in query_lower:
                if relationship.lower() == "brother":
                    score += 0.9

            if "mother" in query_lower:
                if relationship.lower() == "mother":
                    score += 0.9

            if "father" in query_lower:
                if relationship.lower() == "father":
                    score += 0.9

            if "friend" in query_lower:
                if relationship.lower() == "friend":
                    score += 0.4

            content_words = set(
                self._normalize_words(content)
            )

            query_words = set(
                self._normalize_words(query)
            )

            overlap = len(
                content_words & query_words
            )

            if overlap:
                score += min(
                    overlap * 0.08,
                    0.4
                )

            copied = dict(fact)
            copied["_rank_score"] = score

            ranked.append(copied)

        ranked.sort(
            key=lambda x: x.get(
                "_rank_score",
                0.0
            ),
            reverse=True
        )

        return ranked

    async def _extract_query_entities(
        self,
        text: str
    ) -> list[dict]:

        prompt = f"""Identify people or other specific entities explicitly mentioned in the user's message.

User message:
"{text}"

Return ONLY valid JSON.

Format:
[
  {{
    "name": "Rahul Thakur",
    "entity_type": "person"
  }}
]

Rules:
- Only return entities explicitly named in the message.
- Preserve the exact name.
- Do not invent entities.
- A person's name should use entity_type "person".
- Return [] if no explicit named entity exists.
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_logic,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0
            )

            content = (
                response.choices[0].message.content
                or ""
            )

            return self._parse_json_array(content)

        except Exception as e:
            print(
                f"[Query entity extraction error]: {e}"
            )
            return []

    async def _build_memory_context(
        self,
        relevant_facts: list[dict]
    ) -> str:

        if not relevant_facts and not self.personality_traits:
            return ""

        context_parts = []

        context_parts.append(
            "PERSISTENT MEMORY\n"
            "Use this information when it is relevant to the user's message.\n"
            "These are stored facts, not suggestions or guesses.\n"
        )

        if self.personality_traits:
            context_parts.append(
                "PERSONALITY:\n"
            )

            for trait in self.personality_traits[:10]:
                trait_name = str(
                    trait.get("trait_name", "")
                ).strip()

                trait_value = str(
                    trait.get("trait_value", "")
                ).strip()

                if trait_name and trait_value:
                    context_parts.append(
                        f"- {trait_name}: {trait_value}"
                    )

        if relevant_facts:
            context_parts.append(
                "\nFACTS:\n"
            )

            for fact in relevant_facts:
                entity_name = str(
                    fact.get("entity_name", "")
                ).strip()

                relationship = str(
                    fact.get("relationship", "")
                ).strip()

                content = str(
                    fact.get("content", "")
                ).strip()

                if not content:
                    continue

                if entity_name:
                    if relationship:
                        identity = (
                            f"{entity_name} "
                            f"({relationship})"
                        )
                    else:
                        identity = entity_name

                    context_parts.append(
                        f"- [{identity}] {content}"
                    )
                else:
                    context_parts.append(
                        f"- {content}"
                    )

        context_parts.append(
            "\nUse the facts above when they answer the user's question. "
            "Do not ignore relevant stored facts."
        )

        return "\n".join(context_parts)

    async def _extract_and_store_memories(
        self,
        text: str
    ):
        try:
            print("[Extracting memories...]")

            memories = await self._extract_facts(text)

            print(
                f"[Extracted memories]: {memories}"
            )

            if memories:
                await self._store_memories(memories)

            print("[Memory extraction complete]")

        except Exception as e:
            print(
                f"[Memory error]: {e}"
            )

    async def _extract_facts(
        self,
        text: str
    ) -> list[dict]:

        prompt = f"""Extract durable personal memories from this user message.

User message:
"{text}"

Return ONLY valid JSON.

Format:
[
  {{
    "fact": "My favorite color is blue",
    "category": "personal",
    "entity": {{
      "name": "user",
      "entity_type": "user",
      "relationship": "self"
    }},
    "confidence": 1.0,
    "importance": 0.8
  }}
]

Categories:
- name
- work
- education
- skills
- interests
- personal
- relationships
- goals

Rules:
1. Extract only explicit durable information.
2. Do not invent information.
3. First-person facts belong to entity "user".
4. Facts about another person must identify that person.
5. Preserve exact names.
6. Never merge people.
7. Include relationships when explicitly known.
8. Do not store questions.
9. Do not store temporary conversational statements.
10. Do not store instructions such as "remember this" as a fact.
11. Store actual information contained in the instruction.
12. Return [] if there is nothing worth remembering.
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_logic,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0
            )

            content = (
                response.choices[0].message.content
                or ""
            )

            return self._parse_json_array(content)

        except Exception as e:
            print(
                f"[Fact extraction error]: {e}"
            )
            return []

    async def _store_memories(
        self,
        memories: list[dict]
    ):

        for memory in memories:
            fact_text = str(
                memory.get("fact", "")
            ).strip()

            category = str(
                memory.get(
                    "category",
                    "personal"
                )
            ).strip().lower()

            entity_data = (
                memory.get("entity")
                or {}
            )

            entity_name = str(
                entity_data.get(
                    "name",
                    "user"
                )
            ).strip()

            entity_type = str(
                entity_data.get(
                    "entity_type",
                    "user"
                )
            ).strip().lower()

            relationship = entity_data.get(
                "relationship"
            )

            if relationship is not None:
                relationship = str(
                    relationship
                ).strip() or None

            confidence = self._safe_float(
                memory.get(
                    "confidence",
                    1.0
                ),
                1.0
            )

            importance = self._safe_float(
                memory.get(
                    "importance",
                    0.5
                ),
                0.5
            )

            if not fact_text:
                continue

            if entity_type == "user":
                entity_name = "user"
                relationship = "self"

            try:
                entity = (
                    await self.memory_db.get_or_create_entity(
                        name=entity_name,
                        entity_type=entity_type,
                        relationship=relationship
                    )
                )

                entity_id = entity["id"]

                existing_facts = (
                    await self.memory_db
                    .get_facts_for_entity_and_category(
                        entity_id=entity_id,
                        category=category
                    )
                )

                replacement = (
                    await self._find_replacement_fact(
                        fact_text,
                        existing_facts
                    )
                )

                if replacement:
                    print(
                        f"[REPLACING] "
                        f"{replacement['content']} "
                        f"-> {fact_text}"
                    )

                    await self.memory_db.delete_fact(
                        replacement["id"]
                    )

                embedding = await self._embed_text(
                    fact_text
                )

                await self.memory_db.store_fact(
                    content=fact_text,
                    embedding=embedding or [],
                    category=category,
                    entity_id=entity_id,
                    confidence=confidence,
                    importance=importance
                )

                print(
                    f"[MEMORY STORED] "
                    f"{entity_name}: {fact_text}"
                )

            except Exception as e:
                print(
                    f"[Memory store error]: {e}"
                )

    async def _find_replacement_fact(
        self,
        new_fact: str,
        existing_facts: list[dict]
    ) -> Optional[dict]:

        if not existing_facts:
            return None

        existing_texts = "\n".join(
            f"{fact['id']}: {fact['content']}"
            for fact in existing_facts
        )

        prompt = f"""Determine whether the new fact updates or contradicts one of the existing facts.

New fact:
"{new_fact}"

Existing facts:
{existing_texts}

Return ONLY valid JSON.

If it replaces an existing fact:
{{
  "replace": true,
  "fact_id": 123
}}

If it does not:
{{
  "replace": false,
  "fact_id": null
}}

Rules:
- Replace only when the new fact clearly updates or contradicts an existing fact.
- Do not replace unrelated facts.
- Return the exact ID of the fact being replaced.
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_logic,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0
            )

            content = (
                response.choices[0].message.content
                or ""
            )

            data = self._parse_json_object(
                content
            )

            if not data.get("replace"):
                return None

            fact_id = data.get(
                "fact_id"
            )

            if fact_id is None:
                return None

            for fact in existing_facts:
                try:
                    if int(fact["id"]) == int(fact_id):
                        return fact
                except Exception:
                    continue

            return None

        except Exception as e:
            print(
                f"[Contradiction check error]: {e}"
            )
            return None

    async def _embed_text(
        self,
        text: str
    ) -> Optional[list]:

        if not text:
            return None

        try:
            response = await self.client.embeddings.create(
                model=self.config.embedding_model,
                input=text
            )

            return response.data[0].embedding

        except Exception as e:
            print(
                f"[Embedding error]: {e}"
            )
            return None

    async def _update_personality_traits(
        self,
        assistant_response: str
    ):

        if not assistant_response:
            return

        prompt = f"""Identify persistent personality traits demonstrated by Alex in this response.

Response:
"{assistant_response}"

Return ONLY valid JSON.

Format:
[
  {{
    "trait_name": "empathy",
    "trait_value": "shows genuine concern"
  }}
]

Possible traits:
- empathy
- humor
- curiosity
- attention_to_detail
- warmth
- wit

Only return traits clearly demonstrated by the response.

Return [] if there are none.
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_logic,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0
            )

            content = (
                response.choices[0].message.content
                or ""
            )

            traits = self._parse_json_array(
                content
            )

            for trait in traits:
                trait_name = str(
                    trait.get(
                        "trait_name",
                        ""
                    )
                ).strip().lower().replace(
                    " ",
                    "_"
                )

                trait_value = str(
                    trait.get(
                        "trait_value",
                        ""
                    )
                ).strip()

                if not trait_name or not trait_value:
                    continue

                try:
                    await self.memory_db.store_personality_trait(
                        trait_name=trait_name,
                        trait_value=trait_value
                    )
                except Exception as e:
                    print(
                        f"[Personality storage error]: {e}"
                    )

            try:
                self.personality_traits = (
                    await self.memory_db
                    .get_personality_traits()
                )
            except Exception:
                pass

        except Exception as e:
            print(
                f"[Personality extraction error]: {e}"
            )

    def _is_first_person_query(
        self,
        text: str
    ) -> bool:

        text_lower = text.lower().strip()

        first_person_patterns = [
            r"\bmy\b",
            r"\bme\b",
            r"\bi\b",
            r"\bmine\b",
            r"\bmyself\b",
            r"\babout me\b",
            r"\bdo you remember\b"
        ]

        return any(
            re.search(
                pattern,
                text_lower
            )
            for pattern in first_person_patterns
        )

    @staticmethod
    def _same_entity(
        first: str,
        second: str
    ) -> bool:

        if not first or not second:
            return False

        return (
            first.strip().lower()
            ==
            second.strip().lower()
        )

    @staticmethod
    def _normalize_words(
        text: str
    ) -> list[str]:

        return re.findall(
            r"[a-zA-Z0-9']+",
            text.lower()
        )

    @staticmethod
    def _parse_json_array(
        text: str
    ) -> list:

        if not text:
            return []

        try:
            data = json.loads(text)

            if isinstance(data, list):
                return data

        except Exception:
            pass

        match = re.search(
            r"\[[\s\S]*\]",
            text
        )

        if not match:
            return []

        try:
            data = json.loads(
                match.group(0)
            )

            return (
                data
                if isinstance(data, list)
                else []
            )

        except Exception:
            return []

    @staticmethod
    def _parse_json_object(
        text: str
    ) -> dict:

        if not text:
            return {}

        try:
            data = json.loads(text)

            if isinstance(data, dict):
                return data

        except Exception:
            pass

        match = re.search(
            r"\{[\s\S]*\}",
            text
        )

        if not match:
            return {}

        try:
            data = json.loads(
                match.group(0)
            )

            return (
                data
                if isinstance(data, dict)
                else {}
            )

        except Exception:
            return {}

    @staticmethod
    def _safe_float(
        value,
        default: float
    ) -> float:

        try:
            value = float(value)

            if value < 0:
                return 0.0

            if value > 1:
                return 1.0

            return value

        except Exception:
            return default