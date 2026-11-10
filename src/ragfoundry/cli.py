"""Command line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

import typer

from . import RAGPipeline
from .errors import RAGFoundryError

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Build and run RAG pipelines.")

ConfigOption = Annotated[Path, typer.Option("--config", "-c", help="Path to the YAML config.")]

TEMPLATE = """\
# ragfoundry configuration. ${VAR} reads from the environment.
persist_dir: .ragfoundry

chunking:
  strategy: recursive   # fixed | recursive | semantic | llm
  chunk_size: 1000
  chunk_overlap: 200
  length_unit: characters   # characters | tokens

embeddings:
  provider: huggingface   # huggingface | openai | gemini
  model: sentence-transformers/all-MiniLM-L6-v2

vector_store:
  provider: chroma   # chroma | faiss | pinecone | qdrant | weaviate | milvus | pgvector
  collection_name: my_docs

llm:
  provider: ollama   # groq | openai | gemini | claude | ollama
  model: llama3.2

retrieval:
  top_k: 5
  fetch_k: 20
  hybrid:
    enabled: true
  # rerank:
  #   provider: huggingface   # huggingface | cohere
  #   model: cross-encoder/ms-marco-MiniLM-L-6-v2
  query_transform:
    strategies: []   # rewrite, multi_query, hyde
  corrective:
    enabled: false

observability:
  provider: none   # none | langsmith | langfuse
  # api_key: ${LANGSMITH_API_KEY}          # LangSmith
  # public_key: ${LANGFUSE_PUBLIC_KEY}     # Langfuse
  # secret_key: ${LANGFUSE_SECRET_KEY}     # Langfuse
"""


def _fail(error: Exception) -> typer.Exit:
    typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
    return typer.Exit(code=1)


@app.command()
def init(path: Path = Path("rag.yaml")) -> None:
    """Write a starter configuration file."""
    if path.exists():
        raise _fail(FileExistsError(f"{path} already exists"))
    path.write_text(TEMPLATE, encoding="utf-8")
    typer.echo(f"Wrote {path}")


@app.command()
def ingest(
    sources: list[str],
    config: ConfigOption = Path("rag.yaml"),
) -> None:
    """Load, chunk and index files, directories or URLs."""
    try:
        with RAGPipeline.from_yaml(config) as pipeline:
            report = pipeline.ingest(sources)
    except RAGFoundryError as error:
        raise _fail(error) from error
    typer.echo(
        f"Documents: {report.documents_loaded}  Chunks: {report.chunks_total}  "
        f"Added: {report.chunks_added}  Unchanged: {report.chunks_unchanged}  "
        f"Removed: {report.chunks_removed}"
    )
    for failure in report.failures:
        typer.secho(f"Failed: {failure.source}: {failure.error}", fg=typer.colors.YELLOW, err=True)
    if report.failures:
        raise typer.Exit(code=1)


@app.command()
def query(
    question: str,
    config: ConfigOption = Path("rag.yaml"),
    stream: Annotated[bool, typer.Option(help="Print the answer as it is generated.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the full result.")] = False,
) -> None:
    """Answer a question from the indexed documents."""
    try:
        with RAGPipeline.from_yaml(config) as pipeline:
            if stream:
                for text in pipeline.stream(question):
                    typer.echo(text, nl=False)
                typer.echo()
                return
            result = pipeline.query(question)
    except RAGFoundryError as error:
        raise _fail(error) from error
    if as_json:
        typer.echo(json.dumps(result.to_dict(), indent=2, ensure_ascii=False, default=str))
        return
    typer.echo(result.answer)
    for source in result.sources:
        typer.echo(f"  - {source}")


@app.command("eval")
def evaluate(
    dataset: Path,
    config: ConfigOption = Path("rag.yaml"),
    mode: Literal["retrieval", "generation", "all"] = "all",
    backend: Literal["native", "ragas"] = "native",
    output: Annotated[Path | None, typer.Option(help="Write the full report as JSON.")] = None,
) -> None:
    """Evaluate retrieval and generation against a .json or .jsonl dataset."""
    try:
        with RAGPipeline.from_yaml(config) as pipeline:
            report = pipeline.evaluate(dataset, mode=mode, backend=backend)
    except RAGFoundryError as error:
        raise _fail(error) from error
    for name, stats in report.summary.items():
        typer.echo(f"{name:<24} {stats['mean']:.3f}  (n={stats['count']})")
    if output:
        report.save(output)
        typer.echo(f"Report written to {output}")


@app.command("sources")
def list_sources(config: ConfigOption = Path("rag.yaml")) -> None:
    """List indexed sources and their chunk counts."""
    try:
        with RAGPipeline.from_yaml(config) as pipeline:
            for source, count in pipeline.sources().items():
                typer.echo(f"{count:>6}  {source}")
    except RAGFoundryError as error:
        raise _fail(error) from error


@app.command("delete-source")
def delete_source(source: str, config: ConfigOption = Path("rag.yaml")) -> None:
    """Remove one source from the index."""
    try:
        with RAGPipeline.from_yaml(config) as pipeline:
            typer.echo(f"Removed {pipeline.delete_source(source)} chunks")
    except RAGFoundryError as error:
        raise _fail(error) from error
