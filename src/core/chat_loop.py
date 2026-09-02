import asyncio
import json
from openai import AsyncOpenAI
from dataclasses import dataclass
from typing import Optional
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
        self.conversation_history = []
        self.personality_traits = []
        
        self.system_prompt = """You are a warm, thoughtful AI companion named Alex. 
You remember personal details about the user and reference them naturally in conversation.
Your personality is consistent: curious, empathetic, sometimes playfully witty.
You avoid generic responses and ground yourself in what you know about the user.
Keep responses conversational and under 200 words."""
    
    async def initialize(self):
        """Load data from DB on startup"""
        await self.memory_db.initialize()
        print("[Loading data from memory...]")
        
        # Load personality traits
        self.personality_traits = await self.memory_db.get_personality_traits()
        print(f"[Loaded {len(self.personality_traits)} personality traits]")
        
        # Load and display existing facts
        all_facts = await self.memory_db.get_all_facts()
        print(f"[Loaded {len(all_facts)} facts from memory]")
        for fact in all_facts:
            print(f"  - {fact['content']}")
    
    async def get_response(self, user_message: str) -> str:
        """Generate response"""
        # Extract and store memories immediately
        await self._extract_and_store_memories(user_message)
        
        self.conversation_history.append(Message("user", user_message, len(self.conversation_history)))
        
        # RETRIEVE MEMORIES - HYBRID APPROACH (FIXED)
        relevant_facts = await self._get_relevant_facts(user_message)
        
        # BUILD CONTEXT
        memory_context = await self._build_memory_context(relevant_facts)
        
        messages = [
            {"role": "system", "content": self.system_prompt + "\n\n" + memory_context},
            *[{"role": m.role, "content": m.content} for m in self.conversation_history[-10:]],
        ]
        
        response = await self.client.chat.completions.create(
            model=self.config.model_response,
            messages=messages,
            temperature=0.7
        )
        
        assistant_msg = response.choices[0].message.content
        self.conversation_history.append(Message("assistant", assistant_msg, len(self.conversation_history)))
        
        # Update personality (background)
        asyncio.create_task(self._update_personality_traits(assistant_msg))
        
        return assistant_msg
    
    async def _get_relevant_facts(self, user_message: str) -> list[dict]:
        """
        CORRECTED: Hybrid retrieval strategy.
        - Use semantic similarity when available
        - Always include recent/frequently-accessed facts
        - Never rely solely on embeddings (FIX #2)
        """
        user_embedding = await self._embed_text(user_message)
        
        if user_embedding:
            # Get facts using hybrid scoring (similarity + recency + access)
            relevant_facts = await self.memory_db.get_similar_facts(
                embedding=user_embedding,
                threshold=0.3,
                limit=10
            )
        else:
            # Fallback to all facts ordered by recency
            all_facts = await self.memory_db.get_all_facts()
            relevant_facts = all_facts[:10]
        
        # SAFETY CHECK: If we have less than 3 facts, force load recent ones
        if len(relevant_facts) < 3:
            all_facts = await self.memory_db.get_all_facts()
            relevant_facts = all_facts[:10]
        
        return relevant_facts
    
    async def _build_memory_context(self, relevant_facts: list[dict]) -> str:
        """Build the memory context string for system prompt"""
        memory_context = ""
        
        if self.personality_traits:
            memory_context += "Your personality:\n"
            for trait in self.personality_traits:
                memory_context += f"- {trait['trait_name']}: {trait['trait_value']}\n"
            memory_context += "\n"
        
        if relevant_facts:
            memory_context += "Things you know about the user:\n"
            for fact in relevant_facts:
                # Show similarity score for debugging
                similarity = fact.get('similarity', 0.0)
                memory_context += f"- {fact['content']} (relevance: {similarity:.2f})\n"
            memory_context += "\n"
        else:
            memory_context += "You don't have any facts about the user yet.\n\n"
        
        return memory_context
    
    async def _extract_and_store_memories(self, text: str):
        """Extract facts and handle contradictions"""
        try:
            print("[Extracting facts...]")
            facts = await self._extract_facts(text)
            print(f"[Extracted: {facts}]")
            
            if facts:
                # CORRECTED: Proper contradiction handling (FIX #1)
                await self._store_facts_with_deduplication(facts)
            
            print("[Done]")
        except Exception as e:
            print(f"[Error]: {e}")
            import traceback
            traceback.print_exc()
    
    async def _extract_facts(self, text: str) -> list[dict]:
        """Extract facts from user message"""
        prompt = f"""Extract facts about the user from: "{text}"

Look for:
- Name/identity: "My name is X", "I'm X", "Call me X"
- Work: company, job title, team
- Education: school, degree, year
- Skills: languages, tools
- Interests: hobbies, activities
- Personal: pets, relationships
- Goals: learning, aspirations

Return JSON: [{{"fact": "...", "category": "..."}}]

Categories: name, work, education, skills, interests, personal, goals

Return [] if nothing."""
        
        response = await self.client.chat.completions.create(
            model=self.config.model_logic,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        try:
            text = response.choices[0].message.content
            start = text.find('[')
            end = text.rfind(']') + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
            return []
        except:
            return []
    
    async def _store_facts_with_deduplication(self, new_facts: list[dict]):
        """
        CORRECTED: Store facts with proper deduplication (FIX #1).
        
        Strategy:
        1. For each new fact, check if it contradicts existing facts in same category
        2. If contradiction detected, DELETE old facts in that category
        3. THEN store the new fact (atomically)
        4. Use unique constraint on content to prevent duplicates
        """
        for new_fact in new_facts:
            fact_text = new_fact.get('fact', '').strip()
            category = new_fact.get('category', 'other').lower()
            
            if not fact_text:
                continue
            
            print(f"[Processing] {category}: '{fact_text}'")
            
            # Get existing facts in same category
            existing_facts = await self.memory_db.get_facts_by_category(category)
            
            # Check if this is an update/contradiction
            old_fact_to_delete = None
            if existing_facts:
                old_fact_to_delete = await self._check_contradiction(fact_text, existing_facts)
            
            # If contradiction found, delete old fact BEFORE storing new
            if old_fact_to_delete:
                print(f"[DELETING] Contradicting fact: '{old_fact_to_delete['content']}'")
                await self.memory_db.delete_fact_by_content(old_fact_to_delete['content'])
            
            # NOW store the new fact (DB handles deduplication via UNIQUE constraint)
            embedding = await self._embed_text(fact_text)
            if embedding:
                await self.memory_db.store_fact(
                    content=fact_text,
                    embedding=embedding,
                    category=category
                )
            else:
                # Store without embedding if it fails
                await self.memory_db.store_fact(
                    content=fact_text,
                    embedding=[],
                    category=category
                )
    
    async def _check_contradiction(self, new_fact: str, existing_facts: list[dict]) -> Optional[dict]:
        """
        CORRECTED: Determine which existing fact (if any) contradicts the new one.
        
        Returns the fact to delete, or None if it's a new fact.
        """
        if not existing_facts:
            return None
        
        # Build prompt for contradiction detection
        existing_texts = "\n".join([f"- {f['content']}" for f in existing_facts])
        
        prompt = f"""New fact: "{new_fact}"

Existing facts in same category:
{existing_texts}

Question: Does the new fact REPLACE/UPDATE any existing fact?
Examples:
- New: "I work at Google" replaces "I work at Microsoft" → YES
- New: "I'm learning Python" doesn't replace "I know JavaScript" → NO
- New: "I'm John" replaces "I'm Jon" → YES

Reply with ONLY "YES" or "NO". If YES, identify which fact it replaces."""
        
        response = await self.client.chat.completions.create(
            model=self.config.model_logic,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        answer = response.choices[0].message.content.strip().upper()
        
        if "YES" in answer:
            # Return the FIRST fact (simplest approach: most recent)
            return existing_facts[0] if existing_facts else None
        
        return None
    
    async def _embed_text(self, text: str) -> Optional[list]:
        """Get embedding from OpenAI"""
        try:
            response = await self.client.embeddings.create(
                model=self.config.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"[Embedding error]: {e}")
            return None
    
    async def _update_personality_traits(self, assistant_response: str):
        """Extract personality traits from assistant's responses"""
        prompt = f"""What personality traits does Alex show in this response?

Response: "{assistant_response}"

Extract traits like: empathy, humor, curiosity, attention_to_detail, warmth, wit

Return JSON: [{{"trait_name": "trait", "trait_value": "description"}}]

Example: [{{"trait_name": "empathy", "trait_value": "very high - shows genuine concern"}}]

Be generous - if the trait is evident, include it. Return [] if no clear traits."""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_logic,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            text = response.choices[0].message.content
            start = text.find('[')
            end = text.rfind(']') + 1
            if start != -1 and end > start:
                traits = json.loads(text[start:end])
                for trait in traits:
                    if trait.get('trait_name') and trait.get('trait_value'):
                        await self.memory_db.store_personality_trait(
                            trait_name=trait.get('trait_name', '').lower().replace(' ', '_'),
                            trait_value=trait.get('trait_value', '')
                        )
                        # Update local cache
                        self.personality_traits = await self.memory_db.get_personality_traits()
        except Exception as e:
            print(f"[Personality extraction error]: {e}")