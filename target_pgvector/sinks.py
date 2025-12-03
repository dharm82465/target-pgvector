"""PGVector target sink class, which handles writing streams."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import psycopg2
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from psycopg2 import sql
from sentence_transformers import SentenceTransformer
from singer_sdk.sinks import BatchSink
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from singer_sdk.target_base import Target


class TargetPGVector(BatchSink):
    """PGVector target sink class."""

    max_size = 10  # Max records to write in one batch
    connection = None

    converter = DocumentConverter()

    def __init__(
        self, target: Target, stream_name: str, schema: dict, key_properties: list[str] | None
    ) -> None:
        """Initialize the PGVectorSink and establish database connection.

        Args:
            target: The target instance.
            stream_name: Name of the stream.
            schema: JSON schema for the stream.
            key_properties: Primary key properties.
        """
        super().__init__(target, stream_name, schema, key_properties)
        self.embeddings_table = self.config.get("embeddings_table") or os.environ.get(
            "PGVECTOR_EMBEDDINGS_TABLE"
        )
        self.embeddings_model = SentenceTransformer(
            target.config.get("embeddings_model") or os.environ.get("PGVECTOR_EMBEDDINGS_MODEL")
        )
        try:
            conn = psycopg2.connect(
                host=target.config.get("host") or os.environ.get("PGVECTOR_HOST"),
                port=target.config.get("port") or os.environ.get("PGVECTOR_PORT"),
                dbname=target.config.get("database") or os.environ.get("PGVECTOR_DATABASE"),
                user=target.config.get("user") or os.environ.get("PGVECTOR_USER"),
                password=target.config.get("password") or os.environ.get("PGVECTOR_PASSWORD"),
            )
            conn.autocommit = True
            self.connection = conn
            self.chunker = HybridChunker(
                tokenizer=HuggingFaceTokenizer(
                    tokenizer=AutoTokenizer.from_pretrained(
                        self.embeddings_model.model_name_or_path
                    ),
                    merge_peers=True,
                )
            )
        except psycopg2.Error:
            self.logger.exception("Error connecting to database")

    def setup(self) -> None:
        """Set up the sink by creating necessary tables."""
        super().setup()
        if self.connection:
            cursor = self.connection.cursor()
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {self.stream_name} (
                id BIGINT PRIMARY KEY,
                title TEXT,
                type TEXT,
                author TEXT,
                url TEXT,
                last_modified TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS {self.embeddings_table} (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                document_id BIGINT NOT NULL REFERENCES {self.stream_name}(id) ON DELETE CASCADE,
                chunk_index INT NOT NULL,
                chunk_text TEXT NOT NULL,
                metadata JSONB,  -- chunk-level metadata only
                UNIQUE(document_id, chunk_index)
            );
            """
            cursor.execute(create_table_query)
            cursor.close()
            self.logger.info("Table '%s' is set up.", self.stream_name)

    def clean_up(self) -> None:
        """Finalize the sink by closing the database connection."""
        if self.connection:
            self.connection.close()
            self.logger.info("Database connection closed.")
        super().clean_up()

    def start_batch(self, context: dict) -> None:
        """Start a new batch with the given context.

        Args:
            context: Stream partition or context dictionary.
        """
        self.logger.info("Starting new batch: %s", context["batch_id"])

    def process_batch(self, context: dict) -> None:
        """Write out any prepped records and return once fully written.

        Args:
            context: Stream partition or context dictionary.
        """
        records_to_drain = context["records"]
        cur = self.connection.cursor()
        for record in records_to_drain:
            self.logger.info("Processing record: %s", record["title"])
            query = sql.SQL(
                "INSERT INTO {} (id, title, type, author, url, last_modified) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET "
                "title = EXCLUDED.title, type = EXCLUDED.type, "
                "author = EXCLUDED.author, url = EXCLUDED.url, "
                "last_modified = EXCLUDED.last_modified"
            ).format(sql.Identifier(self.stream_name))
            cur.execute(
                query,
                (
                    record["id"],
                    record["title"],
                    record["type"],
                    record["author"],
                    record["url"],
                    record["last_modified"],
                ),
            )
            document = self.converter.convert_string(record["value"], InputFormat.MD)
            chunk_iter = self.chunker.chunk(dl_doc=document.document)
            chunks = list(chunk_iter)
            for i, chunk in enumerate(chunks):
                metadata = {
                    "filename": chunk.meta.origin.filename,
                    "page_numbers": sorted(
                        {prov.page_no for item in chunk.meta.doc_items for prov in item.prov}
                    )
                    or None,
                    "title": chunk.meta.headings[0] if chunk.meta.headings else None,
                }

                insert_query = sql.SQL(
                    "INSERT INTO {} (document_id, chunk_index, chunk_text, metadata) "
                    "VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (document_id, chunk_index) DO UPDATE SET "
                    "chunk_text = EXCLUDED.chunk_text, metadata = EXCLUDED.metadata"
                ).format(sql.Identifier(self.embeddings_table))
                cur.execute(
                    insert_query,
                    (
                        record["id"],
                        i,
                        chunk.text,
                        json.dumps(metadata),
                    ),
                )
        cur.close()
        self.logger.info("Batch %s processed.", context["batch_id"])
