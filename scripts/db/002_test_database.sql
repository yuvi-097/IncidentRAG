-- Executed by the postgres image on first start of an empty data volume, after 001.
-- A separate database for the integration tests (docker compose --profile test run tests):
-- they replace the contents of the OpsRAG tables, so they never run on the application's.
CREATE DATABASE opsrag_test;
\connect opsrag_test
CREATE EXTENSION IF NOT EXISTS vector;
