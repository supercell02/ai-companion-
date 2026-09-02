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
        
        # RETRIEVE RELEVANT MEMORIES
        user_embedding = await self._embed_text(user_message)
        relevant_facts = []
        
        print(f"\n[DEBUG] User message: '{user_message}'")
        print(f"[DEBUG] User embedding generated: {user_embedding is not None}")
        
        if user_embedding:
            relevant_facts = await self.memory_db.get_similar_facts(
                embedding=user_embedding,
                threshold=0.5,  # LOWERED from 0.6
                limit=5
            )
            print(f"[DEBUG] Found {len(relevant_facts)} similar facts")
            for fact in relevant_facts:
                print(f"[DEBUG]   - {fact['content']} (similarity: {fact['similarity']:.2f})")
        
        # BUILD CONTEXT
        memory_context = ""
        
        if self.personality_traits:
            memory_context += "Your personality:\n"
            for trait in self.personality_traits:
                memory_context += f"- {trait['trait_name']}: {trait['trait_value']}\n"
            memory_context += "\n"
        
        if relevant_facts:
            memory_context += "Things you know about the user:\n"
            for fact in relevant_facts:
                memory_context += f"- {fact['content']}\n"
            memory_context += "\n"
        
        print(f"[DEBUG] Memory context:\n{memory_context}\n")
        
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
    async def _extract_and_store_memories(self, text: str):
        """Extract facts and handle contradictions"""
        try:
            print("[Extracting facts...]")
            facts = await self._extract_facts(text)
            print(f"[Extracted: {facts}]")
            
            if facts:
                # Check for contradictions and store
                await self._store_facts_with_contradiction_handling(facts)
            
            print("[Done]")
        except Exception as e:
            print(f"[Error]: {e}")
            import traceback
            traceback.print_exc()
    
    async def _extract_facts(self, text: str) -> list[dict]:
        """Extract facts from user message"""
        prompt = f"""Extract key facts about the user from: "{text}"
Return JSON: [{{"fact": "...", "category": "..."}}, ...]
Categories: work, personal, interests, skills, goals, relationships
Return [] if no facts."""
        
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
    
    async def _store_facts_with_contradiction_handling(self, new_facts: list[dict]):
        """Store facts and handle contradictions"""
        existing_facts = await self.memory_db.get_all_facts()
        
        for new_fact in new_facts:
            fact_text = new_fact.get('fact', '')
            category = new_fact.get('category', 'other')
            
            # Check for contradictions
            contradiction = await self._check_contradiction(fact_text, existing_facts)
            
            if contradiction:
                print(f"[Contradiction detected] Updating: '{contradiction['content']}'")
                # Delete old fact, store new one
                await self.memory_db.delete_fact(contradiction['id'])
            
            # Embed and store
            embedding = await self._embed_text(fact_text)
            if embedding:
                await self.memory_db.store_fact(
                    content=fact_text,
                    embedding=embedding,
                    category=category
                )
    
    async def _check_contradiction(self, new_fact: str, existing_facts: list[dict]) -> Optional[dict]:
        """Check if new fact contradicts existing ones"""
        if not existing_facts:
            return None
        
        existing_texts = "\n".join([f["content"] for f in existing_facts])
        
        prompt = f"""Does this new fact contradict any existing fact?

New fact: "{new_fact}"

Existing facts:
{existing_texts}

Reply ONLY with JSON: {{"contradicts": true/false, "conflicting_fact": "..."}} or {{"contradicts": false}}"""
        
        response = await self.client.chat.completions.create(
            model=self.config.model_logic,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            if result.get('contradicts'):
                # Find the conflicting fact
                conflict_text = result.get('conflicting_fact', '')
                for fact in existing_facts:
                    if conflict_text.lower() in fact['content'].lower():
                        return fact
            return None
        except:
            return None
    
    async def _embed_text(self, text: str) -> Optional[list]:
        """Get embedding"""
        try:
            response = await self.client.embeddings.create(
                model=self.config.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding error: {e}")
            return None
    
    async def _update_personality_traits(self, assistant_response: str):
        """Extract personality traits"""
        prompt = f"""Extract personality traits from: "{assistant_response}"
Return JSON: [{{"trait_name": "...", "trait_value": "..."}}, ...]
Return [] if none."""
        
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
                    await self.memory_db.store_personality_trait(
                        trait_name=trait.get('trait_name', ''),
                        trait_value=trait.get('trait_value', '')
                    )
                    # Update local cache
                    self.personality_traits = await self.memory_db.get_personality_traits()
        except Exception as e:
            print(f"Personality extraction error: {e}")