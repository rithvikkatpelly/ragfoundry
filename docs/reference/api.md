# Python API

Everything below is importable from `ragfoundry` unless a different module is shown.

## Pipeline

::: ragfoundry.RAGPipeline
    options:
      members:
        - from_yaml
        - ingest
        - query
        - aquery
        - stream
        - astream
        - retrieve
        - chat
        - sources
        - delete_source
        - evaluate
        - flush
        - close

::: ragfoundry.RAGResult

## Chat

::: ragfoundry.ChatSession

## Configuration

::: ragfoundry.RAGConfig
    options:
      members:
        - from_yaml

## Ingestion

::: ragfoundry.IngestReport

::: ragfoundry.load_sources

::: ragfoundry.chunk_documents

::: ragfoundry.register_loader

::: ragfoundry.register_chunker

## Context window

::: ragfoundry.pack_documents

::: ragfoundry.window_history

::: ragfoundry.context.approximate_tokens

::: ragfoundry.context.build_token_counter

## Evaluation

::: ragfoundry.EvalSample

::: ragfoundry.EvalReport

::: ragfoundry.load_dataset

## Errors

::: ragfoundry.RAGFoundryError

::: ragfoundry.ConfigurationError

::: ragfoundry.IngestionError

::: ragfoundry.MissingDependencyError
