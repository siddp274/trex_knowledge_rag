# ingestion/graph_store.py
"""
Neo4j database management with auto-creation.

Handles:
- Creating a new Neo4j database if it doesn't exist
- Setting up schema constraints and indexes
- Providing a configured driver for other modules

IMPORTANT: Neo4j database creation requires Enterprise Edition or Aura.
Community Edition only supports the default 'neo4j' database.
If auto_create fails on Community Edition, we log a warning and 
fall back to the default database.
"""
import logging
from neo4j import GraphDatabase
from neo4j.exceptions import ClientError, DatabaseError

from config import TREXConfig

logger = logging.getLogger(__name__)


class Neo4jManager:
    """
    Manages Neo4j database lifecycle and schema setup.
    
    Schema created on init:
    - UNIQUE constraint on Entity.name (prevents exact duplicates)
    - Index on Entity.type (fast filtering by entity type)
    - UNIQUE constraint on ChunkRef.chunk_id (for provenance bridge)
    """

    def __init__(self, config: TREXConfig):
        self.config = config
        self.driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password),
        )
        # Verify connectivity
        self.driver.verify_connectivity()
        logger.info(f"[Neo4j] Connected to {config.neo4j_uri}")

        if config.neo4j_auto_create:
            self._ensure_database()

        self._setup_schema()

    def _ensure_database(self):
        """
        Create the database if it doesn't exist.
        
        Uses the 'system' database to issue CREATE DATABASE.
        This only works on Neo4j Enterprise / Aura.
        On Community Edition, we catch the error and fall back 
        to the default 'neo4j' database.
        """
        target_db = self.config.neo4j_database

        if target_db == "neo4j":
            # Default database always exists, skip creation
            return

        try:
            with self.driver.session(database="system") as session:
                # Check if database already exists
                result = session.run("SHOW DATABASES")
                existing_dbs = {record["name"] for record in result}

                if target_db in existing_dbs:
                    logger.info(
                        f"[Neo4j] Database '{target_db}' already exists — reusing"
                    )
                    return

                # Create the database
                session.run(f"CREATE DATABASE `{target_db}` IF NOT EXISTS")
                logger.info(f"[Neo4j] Created database '{target_db}'")

        except (ClientError, DatabaseError) as e:
            error_msg = str(e)
            if "Unsupported administration command" in error_msg or \
               "Unable to route write" in error_msg or \
               "administrative" in error_msg.lower():
                logger.warning(
                    f"[Neo4j] Cannot create database '{target_db}' — "
                    f"likely running Community Edition (single-database only). "
                    f"Falling back to default 'neo4j' database."
                )
                self.config.neo4j_database = "neo4j"
            else:
                raise

    def _setup_schema(self):
        """
        Create constraints and indexes if they don't exist.
        
        These are idempotent (IF NOT EXISTS) so safe to run 
        on every startup.
        """
        db = self.config.neo4j_database
        with self.driver.session(database=db) as session:
            # Entity name must be unique — this is how we MERGE by name
            session.run(
                "CREATE CONSTRAINT entity_name IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )
            # Fast lookup by entity type
            session.run(
                "CREATE INDEX entity_type IF NOT EXISTS "
                "FOR (e:Entity) ON (e.type)"
            )
            # ChunkRef nodes link entities back to vector-stored chunks
            session.run(
                "CREATE CONSTRAINT chunk_ref_id IF NOT EXISTS "
                "FOR (c:ChunkRef) REQUIRE c.chunk_id IS UNIQUE"
            )
            # Full-text index for fuzzy entity search at query time
            try:
                session.run(
                    "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS "
                    "FOR (e:Entity) ON EACH [e.name, e.description]"
                )
            except ClientError:
                # Full-text indexes have different syntax across Neo4j versions
                logger.warning("[Neo4j] Full-text index creation failed — skipping")

        logger.info(f"[Neo4j] Schema ready on database '{db}'")

    def get_session(self, **kwargs):
        """Get a session on the configured database."""
        return self.driver.session(database=self.config.neo4j_database, **kwargs)

    def close(self):
        """Close the driver connection."""
        self.driver.close()