CREATE TABLE IF NOT EXISTS facts (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    category VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    user_message TEXT,
    assistant_response TEXT,
    turn_number INT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS personality_traits (
    id SERIAL PRIMARY KEY,
    trait_name VARCHAR(100),
    trait_value TEXT,
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_facts_embedding ON facts USING ivfflat (embedding vector_cosine_ops);