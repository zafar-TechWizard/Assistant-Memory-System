# Contributing

## Dev environment setup

1. Install prerequisites: Python 3.11+, [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Clone and install in editable mode:
   ```bash
   git clone https://github.com/your-org/assistant-memory
   cd assistant-memory
   pip install -e ".[nlp,dev]"
   python -m spacy download en_core_web_sm
   python -m coreferee install en
   ```
3. Copy `.env.example` to `.env` and fill in your values.

## Running tests

```bash
pytest
```

Integration tests that require a live Docker/Neo4j instance are marked with
`@pytest.mark.integration` and are skipped automatically when Docker is unavailable.

## Branch model

- `prod` — stable release branch; all PRs target here
- `release/v-*` — release staging branches
- Feature branches: `feat/<name>`, bugfix branches: `fix/<name>`

## Code conventions

- **Type hints** on all public functions and class methods.
- **Async patterns**: always use `get_context_async()` from async callers — never `get_context()` (sync) from inside `async def`.
- **No blocking I/O on the event loop**: wrap `subprocess.run`, file I/O, and HTTP calls in `run_in_executor` when inside `async def`.
- **Docstrings** on all public classes and methods. One-line docstrings for private helpers.
- **No hardcoded identity**: use `config.assistant_name` / `config.user_id`; never hardcode names.

## Pull request checklist

- [ ] Tests pass (`pytest`)
- [ ] No new blocking I/O in async paths
- [ ] No hardcoded credentials or personal identifiers
- [ ] Public API changes reflected in `__init__.py` and docstrings
