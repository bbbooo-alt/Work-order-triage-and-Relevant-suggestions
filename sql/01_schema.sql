CREATE EXTENSION IF NOT EXISTS vector;

-- 数据层
CREATE TABLE IF NOT EXISTS skill_domains (
    domain_id  smallint PRIMARY KEY,
    name       text NOT NULL UNIQUE,
    aliases    text[] NOT NULL DEFAULT '{}',
    parent_id  smallint REFERENCES skill_domains(domain_id)
);

CREATE TABLE IF NOT EXISTS engineers (
    engineer_id smallint PRIMARY KEY,
    name        text NOT NULL,
    team        text NOT NULL,
    level       smallint NOT NULL CHECK (level BETWEEN 1 AND 3),
    capacity    smallint NOT NULL CHECK (capacity > 0),
    timezone    text NOT NULL DEFAULT 'Asia/Shanghai',
    active      boolean NOT NULL DEFAULT true
);

CREATE TABLE IF NOT EXISTS tickets (
    ticket_id       integer PRIMARY KEY,
    subject         text,
    body            text NOT NULL,
    body_zh         text,
    answer          text,
    language        text NOT NULL CHECK (language IN ('en', 'de')),
    tags            text[] NOT NULL DEFAULT '{}',
    type            text NOT NULL CHECK (type IN ('Incident', 'Request', 'Problem', 'Change')),
    queue           text NOT NULL,
    priority        text NOT NULL CHECK (priority IN ('low', 'medium', 'high')),
    impact          smallint CHECK (impact BETWEEN 1 AND 3),
    urgency         smallint CHECK (urgency BETWEEN 1 AND 3),
    required_skills integer[] NOT NULL DEFAULT '{}',
    status          text NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'pending', 'closed')),
    created_at      timestamptz NOT NULL,
    sla_due         timestamptz,
    resolved_at     timestamptz,
    assignee_id     smallint REFERENCES engineers(engineer_id),
    split           text NOT NULL CHECK (split IN ('train', 'val', 'test')),
    source_file     text NOT NULL
);

CREATE TABLE IF NOT EXISTS resolution_knowledge (
    chunk_id    bigserial PRIMARY KEY,
    ticket_id   integer NOT NULL REFERENCES tickets(ticket_id),
    chunk_text  text NOT NULL,
    chunk_type  text NOT NULL,
    embedding   vector(1024),
    domain_id   smallint REFERENCES skill_domains(domain_id)
);

-- 资源层
CREATE TABLE IF NOT EXISTS engineer_skills (
    engineer_id smallint NOT NULL REFERENCES engineers(engineer_id),
    domain_id   smallint NOT NULL REFERENCES skill_domains(domain_id),
    proficiency smallint NOT NULL CHECK (proficiency BETWEEN 1 AND 5),
    PRIMARY KEY (engineer_id, domain_id)
);

CREATE TABLE IF NOT EXISTS shifts (
    shift_id    bigserial PRIMARY KEY,
    engineer_id smallint NOT NULL REFERENCES engineers(engineer_id),
    shift_start timestamptz NOT NULL,
    shift_end   timestamptz NOT NULL,
    on_call     boolean NOT NULL DEFAULT false,
    timezone    text NOT NULL DEFAULT 'Asia/Shanghai',
    CHECK (shift_end > shift_start)
);

-- 决策层（冷启动保持为空，由系统运行写入）
CREATE TABLE IF NOT EXISTS triage_results (
    result_id     bigserial PRIMARY KEY,
    ticket_id     integer NOT NULL REFERENCES tickets(ticket_id),
    model_version text NOT NULL,
    pred_type     text,
    pred_queue    text,
    pred_priority text,
    confidence    real,
    pred_skills   integer[] NOT NULL DEFAULT '{}',
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS assignments (
    assignment_id  bigserial PRIMARY KEY,
    ticket_id      integer NOT NULL REFERENCES tickets(ticket_id),
    engineer_id    smallint NOT NULL REFERENCES engineers(engineer_id),
    decided_at     timestamptz NOT NULL DEFAULT now(),
    status         text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'accepted', 'closed')),
    load_at_assign smallint NOT NULL,
    reason         text,
    closed_at      timestamptz
);
