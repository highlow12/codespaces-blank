# Agent instructions

## Search tools

- For text and file searches in this workspace, always use `./.local/bin/rg` (or the absolute path `/workspaces/codespaces-blank/.local/bin/rg`) rather than relying on `rg` being available on `PATH`.
- Use `./.local/bin/rg --files` when listing tracked or unignored files, and `./.local/bin/rg <pattern>` when searching file contents.

## Python environment

- Use the repository virtual environment at `./.venv/bin/python` for scripts, imports, and tests.
- If it does not exist, create it with `python3 -m venv .venv`.
- Install dependencies with `./.venv/bin/python -m pip install -r requirements.txt`.
- Run the test suite with `./.venv/bin/python -m unittest discover -s . -p 'test_*.py'`.
