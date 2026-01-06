"""PGVector target sink class, which handles writing streams."""

from __future__ import annotations

import json
import os
from multiprocessing import context
from typing import TYPE_CHECKING

import psycopg2
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from openai import OpenAI
from psycopg2 import sql
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
        self.document_stream_name = self.config.get("document_stream_name") or os.environ.get(
            "PGVECTOR_DOCUMENT_STREAM_NAME"
        )
        self.embeddings_model_name = self.config.get("embeddings_model") or os.environ.get(
            "PGVECTOR_EMBEDDINGS_MODEL"
        )

        self.document_text_properties = self.config.get(
            "document_text_properties"
        ) or os.environ.get("PGVECTOR_DOCUMENT_TEXT_PROPERTIES")

        self.embedding_dimension = self.config.get("embeddings_dimension") or os.environ.get(
            "PGVECTOR_EMBEDDINGS_DIMENSION"
        )

        self.openaiclient = OpenAI(
            base_url=self.config.get("openai_base_url")
            or os.environ.get("PGVECTOR_OPENAI_BASE_URL"),
            api_key=self.config.get("openai_api_key") or os.environ.get("PGVECTOR_OPENAI_API_KEY"),
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
                        self.embeddings_model_name
                        if self.embeddings_model_name != "all-mpnet-base-v2"
                        else "sentence-transformers/all-mpnet-base-v2"
                    ),
                    merge_peers=True,
                )
            )
        except psycopg2.Error:
            self.logger.exception("Error connecting to database")

    def setup(self) -> None:
        """Set up the sink by creating necessary tables based on schema."""
        super().setup()
        if self.connection and self.document_stream_name == self.stream_name:
            cursor = self.connection.cursor()

            # Build columns from schema
            columns = self._build_columns_from_schema()

            # Create main table
            create_table_query = sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
                sql.Identifier(self.stream_name), sql.SQL(", ").join(columns)
            )
            cursor.execute(create_table_query)

            create_embeddings_query = sql.SQL("""
                CREATE TABLE IF NOT EXISTS {} (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id INT NOT NULL REFERENCES {}(id) ON DELETE CASCADE,
                    chunk_index INT NOT NULL,
                    chunk_text TEXT NOT NULL,
                    metadata JSONB,
                    embeddings vector({}),
                    UNIQUE(document_id, chunk_index)
                )
            """).format(
                sql.Identifier(self.document_stream_name + "_embeddings"),
                sql.Identifier(self.stream_name),
                sql.Literal(self.embedding_dimension),
            )
            cursor.execute(create_embeddings_query)

            # Create index
            create_index_query = sql.SQL(
                "CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw (embeddings vector_l2_ops)"
            ).format(
                sql.Identifier(f"{self.document_stream_name}_embeddings_idx"),
                sql.Identifier(self.document_stream_name + "_embeddings"),
            )
            cursor.execute(create_index_query)

            cursor.close()
            self.logger.info("Tables created for stream '%s'", self.stream_name)
        else:
            self.logger.warning(
                "Skipping table creation for stream '%s' as it does not match document_stream_name",
                self.stream_name,
            )

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
        """Write records dynamically based on schema."""
        records_to_drain = context["records"]
        cur = self.connection.cursor()

        for record in records_to_drain:
            # Build dynamic insert
            fields = list(record.keys())
            values = [record[f] for f in fields]

            if len(self.key_properties) > 0:
                query = sql.SQL(
                    "INSERT INTO {} ({}) VALUES ({}) ON CONFLICT ({}) DO UPDATE SET {}"
                ).format(
                    sql.Identifier(self.stream_name),
                    sql.SQL(", ").join(sql.Identifier(f) for f in fields),
                    sql.SQL(", ").join(sql.Placeholder() * len(fields)),
                    sql.SQL(", ").join(sql.Identifier(k) for k in self.key_properties),
                    sql.SQL(", ").join(
                        sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(f), sql.Identifier(f))
                        for f in fields
                        if f not in self.key_properties
                    ),
                )
            else:
                query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                    sql.Identifier(self.stream_name),
                    sql.SQL(", ").join(sql.Identifier(f) for f in fields),
                    sql.SQL(", ").join(sql.Placeholder() * len(fields)),
                )

            cur.execute(query, values)

            # Process document chunks and embeddings

            text = "<br/>".join(
                f"<h1>{prop.capitalize()}</h1>\n<p>{record[prop]}</p>" if record[prop] else ""
                for prop in self.document_text_properties
                if prop in record
            )

            document = self.converter.convert_string(text, InputFormat.HTML)
            chunk_iter = self.chunker.chunk(dl_doc=document.document)
            chunks = list(chunk_iter)
            for i, chunk in enumerate(chunks):
                metadata = {
                    "page_numbers": sorted(
                        {prov.page_no for item in chunk.meta.doc_items for prov in item.prov}
                    )
                    or None,
                    "title": chunk.meta.headings[0] if chunk.meta.headings else None,
                }

                insert_query = sql.SQL(
                    "INSERT INTO {} (document_id, chunk_index, chunk_text, metadata, embeddings) "
                    "VALUES (%s, %s, %s, %s, %s) "
                    "ON CONFLICT (document_id, chunk_index) DO UPDATE SET "
                    "chunk_text = EXCLUDED.chunk_text, metadata = EXCLUDED.metadata"
                ).format(sql.Identifier(self.document_stream_name + "_embeddings"))
                cur.execute(
                    insert_query,
                    (
                        record["id"],
                        i,
                        chunk.text,
                        json.dumps(metadata),
                        json.dumps(
                            self.openaiclient.embeddings.create(
                                model=self.embeddings_model_name, input=chunk.text
                            )
                            .data[0]
                            .embedding
                        ),
                    ),
                )
        cur.close()
        self.logger.info("Batch %s processed.", context["batch_id"])

    def _build_columns_from_schema(self) -> list:
        """Build PostgreSQL column definitions from JSON schema.

        Returns:
            List of SQL column definitions.
        """
        columns = []
        properties = self.schema.get("properties", {})

        for field_name, field_schema in properties.items():
            pg_type = self._jsonschema_type_to_postgres(field_schema)
            is_nullable = self._is_nullable(field_schema)
            is_key = field_name in self.key_properties

            column_def = sql.SQL("{} {}").format(sql.Identifier(field_name), sql.SQL(pg_type))

            if is_key:
                column_def = sql.SQL("{} PRIMARY KEY").format(column_def)
            elif not is_nullable:
                column_def = sql.SQL("{} NOT NULL").format(column_def)

            columns.append(column_def)

        return columns

    def _jsonschema_type_to_postgres(self, field_schema: dict) -> str:
        """Convert JSON Schema type to PostgreSQL type.

        Args:
            field_schema: The JSON schema for a field.

        Returns:
            PostgreSQL type string.
        """
        json_type = field_schema.get("type", [])

        # Handle arrays like ["string", "null"]
        if isinstance(json_type, list):
            json_type = [t for t in json_type if t != "null"]
            json_type = json_type[0] if json_type else "string"

        format_type = field_schema.get("format")

        # Map JSON Schema types to PostgreSQL types
        if format_type == "date-time":
            return "TIMESTAMP"
        elif format_type == "date":
            return "DATE"
        elif format_type == "time":
            return "TIME"
        elif json_type == "integer":
            return "BIGINT"
        elif json_type == "number":
            return "NUMERIC"
        elif json_type == "boolean":
            return "BOOLEAN"
        elif json_type == "object" or json_type == "array":
            return "JSONB"
        else:
            return "TEXT"

    def _is_nullable(self, field_schema: dict) -> bool:
        """Check if a field is nullable.

        Args:
            field_schema: The JSON schema for a field.

        Returns:
            True if the field is nullable.
        """
        json_type = field_schema.get("type", [])

        if isinstance(json_type, list):
            return "null" in json_type

        return False
