#!/bin/bash
python -c "import toml; print(toml.load('/workspaces/gai-rag-svr/pyproject.toml')['project']['version'])"
python main.py