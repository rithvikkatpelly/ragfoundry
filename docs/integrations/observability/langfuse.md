# Langfuse

[Langfuse](https://langfuse.com) is an open-source LLM tracing platform, available as a cloud
service or self-hosted.

| | |
|---|---|
| Extra | `langfuse` |
| Keys | A public and a secret key from your Langfuse project settings |

## Install

```bash
pip install "ragfoundry[langfuse]"
```

## Configure

=== "Python"

    ```python
    import os

    "observability": {
        "provider": "langfuse",
        "public_key": os.environ["LANGFUSE_PUBLIC_KEY"],
        "secret_key": os.environ["LANGFUSE_SECRET_KEY"],
        "endpoint": "https://cloud.langfuse.com",
    }
    ```

=== "YAML"

    ```yaml
    observability:
      provider: langfuse
      public_key: ${LANGFUSE_PUBLIC_KEY}
      secret_key: ${LANGFUSE_SECRET_KEY}
      endpoint: https://cloud.langfuse.com
    ```

## Options

| Setting | Meaning |
|---|---|
| `public_key` | Required (`pk-lf-...`) |
| `secret_key` | Required (`sk-lf-...`) |
| `endpoint` | `https://cloud.langfuse.com` (EU), `https://us.cloud.langfuse.com` (US), or your self-hosted URL |

## Self-hosting

Langfuse can run on your own infrastructure with Docker; see the
[Langfuse self-hosting guide](https://langfuse.com/self-hosting). Set `endpoint` to its URL.
