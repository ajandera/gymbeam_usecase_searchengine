CREATE DATABASE database_dev;
GRANT ALL PRIVILEGES ON DATABASE database_dev TO postgres;

CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    category TEXT NOT NULL,
    embedding BYTEA
);