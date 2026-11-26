# LangSmith

[LangSmith](https://smith.langchain.com) is LangChain's tracing and evaluation platform.

| | |
|---|---|
| Extra | `langsmith` |
| Key | A LangSmith API key ([create one](https://smith.langchain.com/settings)) |

## Install

```bash
pip install "ragfoundry[langsmith]"
```

## Configure

=== "Python"

    ```python
    import os

    "observability": {
        "provider": "langsmith",
        "api_key": os.environ["LANGSMITH_API_KEY"],
        "project": "my-rag",
        "tags": ["dev"],
    }
    ```

=== "YAML"

    ```yaml
    observability:
      provider: langsmith
      api_key: ${LANGSMITH_API_KEY}
      project: my-rag
      tags: [dev]
    ```

The `api_key` setting is required; ragfoundry doesn't read it from the environment on its own,
so reference the variable as shown.

## Options

| Setting | Meaning |
|---|---|
| `api_key` | Required |
| `project` | Project the traces go to (created automatically) |
| `endpoint` | API URL for the EU region or a self-hosted LangSmith |
| `tags` | Labels on every trace |

Open your project at [smith.langchain.com](https://smith.langchain.com) to see each
`ragfoundry.query` trace and the steps inside it.
