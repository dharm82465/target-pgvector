"""PGVector target class."""

from __future__ import annotations

from singer_sdk import typing as th
from singer_sdk.target_base import Target

from target_pgvector.sinks import TargetPGVector


class TargetPGVector(Target):
    """Target for PGVector."""

    name = "target-pgvector"

    config_jsonschema = th.PropertiesList(
        th.Property(
            "host",
            th.StringType(nullable=False),
            secret=False,
            required=True,
            title="Host",
            description="The host address for the PGVector database",
        ),
        th.Property(
            "port",
            th.IntegerType(nullable=False),
            secret=False,
            required=True,
            title="Port",
            description="The port number for the PGVector database",
        ),
        th.Property(
            "user",
            th.StringType(nullable=False),
            secret=False,
            required=True,
            title="User",
            description="The user name for the PGVector database",
        ),
        th.Property(
            "password",
            th.StringType(nullable=False),
            secret=True,
            title="Password",
            description="The password for the PGVector database",
        ),
        th.Property(
            "database",
            th.StringType(nullable=False),
            secret=False,
            required=True,
            title="Database",
            description="The database name for the PGVector database",
        ),
        th.Property(
            "embeddings_table",
            th.StringType(nullable=False),
            secret=False,
            required=True,
            title="Embeddings Table",
            description="The embeddings table name for the PGVector database",
        ),
        th.Property(
            "hf_token",
            th.StringType(nullable=False),
            secret=True,
            title="Hugging Face Token",
            description="The Hugging Face token for accessing embeddings",
        ),
        th.Property(
            "embeddings_model",
            th.StringType(nullable=False),
            secret=False,
            required=True,
            title="Hugging Face Model",
            description="The Hugging Face model for generating embeddings",
        ),
    ).to_dict()

    default_sink_class = TargetPGVector


if __name__ == "__main__":
    TargetPGVector.cli()
