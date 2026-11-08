CREATE INDEX IF NOT EXISTS idx_tickets_queue_priority ON tickets (queue, priority);
CREATE INDEX IF NOT EXISTS idx_tickets_type_language ON tickets (type, language);
CREATE INDEX IF NOT EXISTS idx_tickets_split ON tickets (split);
CREATE INDEX IF NOT EXISTS idx_tickets_created_at ON tickets (created_at);
CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets (status);
CREATE INDEX IF NOT EXISTS idx_tickets_tags_gin ON tickets USING GIN (tags);
CREATE INDEX IF NOT EXISTS idx_tickets_required_skills_gin ON tickets USING GIN (required_skills);

CREATE INDEX IF NOT EXISTS idx_engineer_skills_domain ON engineer_skills (domain_id);
CREATE INDEX IF NOT EXISTS idx_shifts_engineer_time ON shifts (engineer_id, shift_start, shift_end);
CREATE INDEX IF NOT EXISTS idx_assignments_engineer_status ON assignments (engineer_id, status);
CREATE INDEX IF NOT EXISTS idx_assignments_ticket ON assignments (ticket_id);
CREATE INDEX IF NOT EXISTS idx_triage_ticket ON triage_results (ticket_id);

CREATE INDEX IF NOT EXISTS idx_rk_ticket ON resolution_knowledge (ticket_id);
CREATE INDEX IF NOT EXISTS idx_rk_domain ON resolution_knowledge (domain_id);
CREATE INDEX IF NOT EXISTS idx_rk_embedding_hnsw ON resolution_knowledge USING HNSW (embedding vector_cosine_ops);
