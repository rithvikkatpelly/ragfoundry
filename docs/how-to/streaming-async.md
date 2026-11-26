# Streaming and async

## Stream an answer

`stream()` yields the answer as the LLM writes it, so users see text immediately instead of
waiting for the whole answer:

```python
for token in rag.stream("Summarise the security policy"):
    print(token, end="", flush=True)
```

In a conversation:

```python
chat = rag.chat()
for token in chat.stream("What does the policy cover?"):
    print(token, end="", flush=True)
```

The turn is added to the chat history once the stream finishes. If nothing relevant is found, the
abstain message is yielded as a single piece.

!!! note
    Streaming yields only text. When you also need the sources, use `query()` / `chat.ask()`, or
    call `rag.retrieve(question)` separately.

## Async

Every answering method has an async version, for web servers and other `asyncio` code:

| Sync | Async |
|---|---|
| `rag.query(q)` | `await rag.aquery(q)` |
| `rag.stream(q)` | `async for t in rag.astream(q)` |
| `chat.ask(q)` | `await chat.aask(q)` |
| `chat.stream(q)` | `async for t in chat.astream(q)` |

```python
import asyncio


async def main():
    results = await asyncio.gather(
        rag.aquery("What is the refund policy?"),
        rag.aquery("When is the office open?"),
    )
    for result in results:
        print(result.answer)

asyncio.run(main())
```

## Streaming in web frameworks

=== "FastAPI"

    ```python
    from fastapi.responses import StreamingResponse

    @app.get("/ask")
    async def ask(q: str):
        return StreamingResponse(rag.astream(q), media_type="text/plain")
    ```

=== "Streamlit"

    ```python
    answer = st.write_stream(st.session_state.chat.stream(question))
    ```

=== "Jupyter / Colab"

    ```python
    for token in rag.stream(question):
        print(token, end="", flush=True)
    ```
