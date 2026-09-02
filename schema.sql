
CREATE EXTENSION IF NOT EXISTS vector;


CREATE TABLE IF NOT EXISTS entities (
    id SERIAL PRIMARY KEY,

    name VARCHAR(255) NOT NULL,

    entity_type VARCHAR(50) NOT NULL DEFAULT 'person',

    relationship VARCHAR(100),

    description TEXT,

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE (name, entity_type)
);

CREATE INDEX IF NOT EXISTS idx_entities_name
    ON entities(name);

CREATE INDEX IF NOT EXISTS idx_entities_type
    ON entities(entity_type);

CREATE INDEX IF NOT EXISTS idx_entities_relationship
    ON entities(relationship);


CREATE TABLE IF NOT EXISTS facts (
    id SERIAL PRIMARY KEY,

    entity_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,

    content TEXT NOT NULL,

    category VARCHAR(50) NOT NULL,

    embedding vector(1536),

    confidence REAL DEFAULT 1.0,

    importance REAL DEFAULT 0.5,

    access_count INTEGER DEFAULT 0,

    created_at TIMESTAMP DEFAULT NOW(),

    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_facts_entity
    ON facts(entity_id);

CREATE INDEX IF NOT EXISTS idx_facts_category
    ON facts(category);

CREATE INDEX IF NOT EXISTS idx_facts_created_at
    ON facts(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_facts_updated_at
    ON facts(updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_facts_embedding
    ON facts USING ivfflat (embedding vector_cosine_ops);



CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,

    user_message TEXT NOT NULL,

    assistant_response TEXT NOT NULL,

    turn_number INTEGER,

    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conversations_created_at
    ON conversations(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_conversations_turn
    ON conversations(turn_number);

CREATE TABLE IF NOT EXISTS personality_traits (
    id SERIAL PRIMARY KEY,

    trait_name VARCHAR(100) NOT NULL,

    trait_value TEXT NOT NULL,

    confidence REAL DEFAULT 1.0,

    created_at TIMESTAMP DEFAULT NOW(),

    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(trait_name)
);

CREATE INDEX IF NOT EXISTS idx_personality_trait_name
    ON personality_traits(trait_name);



CREATE TABLE IF NOT EXISTS entity_relationships (
    id SERIAL PRIMARY KEY,

    source_entity_id INTEGER NOT NULL
        REFERENCES entities(id) ON DELETE CASCADE,

    relationship_type VARCHAR(100) NOT NULL,

    target_entity_id INTEGER NOT NULL
        REFERENCES entities(id) ON DELETE CASCADE,

    confidence REAL DEFAULT 1.0,

    created_at TIMESTAMP DEFAULT NOW(),

    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE (
        source_entity_id,
        relationship_type,
        target_entity_id
    )
);

CREATE INDEX IF NOT EXISTS idx_relationship_source
    ON entity_relationships(source_entity_id);

CREATE INDEX IF NOT EXISTS idx_relationship_target
    ON entity_relationships(target_entity_id);

CREATE INDEX IF NOT EXISTS idx_relationship_type
    ON entity_relationships(relationship_type);



CREATE TABLE IF NOT EXISTS memory_access_log (
    id SERIAL PRIMARY KEY,

    fact_id INTEGER NOT NULL
        REFERENCES facts(id) ON DELETE CASCADE,

    query TEXT,

    similarity REAL,

    accessed_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_memory_access_fact
    ON memory_access_log(fact_id);

CREATE INDEX IF NOT EXISTS idx_memory_access_time
    ON memory_access_log(accessed_at DESC);



CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;


DROP TRIGGER IF EXISTS update_entities_updated_at
    ON entities;

CREATE TRIGGER update_entities_updated_at
BEFORE UPDATE ON entities
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();


DROP TRIGGER IF EXISTS update_facts_updated_at
    ON facts;

CREATE TRIGGER update_facts_updated_at
BEFORE UPDATE ON facts
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();


DROP TRIGGER IF EXISTS update_personality_traits_updated_at
    ON personality_traits;

CREATE TRIGGER update_personality_traits_updated_at
BEFORE UPDATE ON personality_traits
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();


DROP TRIGGER IF EXISTS update_entity_relationships_updated_at
    ON entity_relationships;

CREATE TRIGGER update_entity_relationships_updated_at
BEFORE UPDATE ON entity_relationships
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();


CREATE OR REPLACE VIEW memory_facts AS
SELECT
    f.id,
    f.content,
    f.category,
    f.confidence,
    f.importance,
    f.access_count,
    f.created_at,
    f.updated_at,

    e.id AS entity_id,
    e.name AS entity_name,
    e.entity_type,
    e.relationship

FROM facts f

LEFT JOIN entities e
    ON f.entity_id = e.id;
