CREATE TABLE IF NOT EXISTS checkpoint (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    checkpoint_data JSONB NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);