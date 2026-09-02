"""Test scenarios for evaluation harness"""

TEST_SCENARIOS = [
    {
        "name": "Basic Memory Recall",
        "turns": [
            {
                "user": "I work at Google on the Cloud team",
                "expected_facts": ["Google", "Cloud team"],
                "check_type": "store"
            },
            {
                "user": "Tell me what you know about my work",
                "expected_recall": ["Google", "Cloud team"],
                "check_type": "recall"
            }
        ]
    },
    {
        "name": "Contradiction Handling",
        "turns": [
            {
                "user": "I work at Google",
                "expected_facts": ["Google"],
                "check_type": "store"
            },
            {
                "user": "Actually, I work at Apple now",
                "expected_facts": ["Apple"],
                "check_type": "update"
            },
            {
                "user": "Where do I work?",
                "expected_recall": ["Apple"],
                "not_recall": ["Google"],
                "check_type": "recall"
            }
        ]
    },
    {
        "name": "Multiple Facts",
        "turns": [
            {
                "user": "I'm a software engineer at Google working on AI. I have a golden retriever named Max. I love Python and competitive programming.",
                "expected_facts": ["Google", "AI", "golden retriever", "Max", "Python", "competitive programming"],
                "check_type": "store"
            },
            {
                "user": "Tell me about my hobbies and interests",
                "expected_recall": ["competitive programming", "Python"],
                "check_type": "recall"
            },
            {
                "user": "What about my pet?",
                "expected_recall": ["golden retriever", "Max"],
                "check_type": "recall"
            }
        ]
    },
    {
        "name": "Long-range Consistency (20 turns)",
        "turns": [
            {
                "user": "Hi, I'm Amar. I work at Google and I love coding.",
                "expected_facts": ["Amar", "Google"],
                "check_type": "store"
            },
            {"user": "What's the weather like?", "check_type": "chat"},
            {"user": "Tell me a joke", "check_type": "chat"},
            {"user": "What's your favorite movie genre?", "check_type": "chat"},
            {"user": "How do I learn Python?", "check_type": "chat"},
            {"user": "What time is it?", "check_type": "chat"},
            {"user": "Recommend a book", "check_type": "chat"},
            {"user": "What's machine learning?", "check_type": "chat"},
            {"user": "Do you like hiking?", "check_type": "chat"},
            {"user": "What's your favorite food?", "check_type": "chat"},
            {
                "user": "Remember who I am?",
                "expected_recall": ["Amar", "Google"],
                "check_type": "recall"
            },
            {"user": "Tell me about climate change", "check_type": "chat"},
            {"user": "What's quantum computing?", "check_type": "chat"},
            {"user": "How do I stay healthy?", "check_type": "chat"},
            {"user": "What's your opinion on AI ethics?", "check_type": "chat"},
            {"user": "Explain blockchain", "check_type": "chat"},
            {
                "user": "Where do I work and what's my name?",
                "expected_recall": ["Amar", "Google"],
                "check_type": "recall"
            }
        ]
    },
    {
        "name": "Personality Consistency",
        "turns": [
            {
                "user": "I'm a very detail-oriented person and I love precision",
                "expected_personality": ["detail-oriented", "precision"],
                "check_type": "personality"
            },
            {"user": "Tell me about yourself", "check_type": "chat"},
            {"user": "What kind of companion are you?", "check_type": "chat"},
            {
                "user": "Describe your personality",
                "expected_personality": ["detail-oriented", "precision"],
                "not_personality": ["casual", "vague"],
                "check_type": "personality"
            }
        ]
    }
]