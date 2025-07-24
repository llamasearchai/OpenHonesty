#!/bin/bash

# Set up environment
export PATH="$HOME/Library/Python/3.9/bin:$PATH"
cd "$(dirname "$0")/backend"

echo "Running comprehensive test suite..."
echo "=================================="

# Install dependencies if needed
echo "Installing dependencies..."
poetry install

# Run pytest
echo "Running pytest..."
poetry run pytest tests/ --cov=honesty --cov-report=term-missing -v

# Run linting
echo "Running flake8..."
poetry run flake8 honesty tests --count --select=E9,F63,F7,F82 --show-source --statistics
poetry run flake8 honesty tests --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics

# Run formatting check
echo "Running black check..."
poetry run black honesty tests --check --diff

# Run import sorting check
echo "Running isort check..."
poetry run isort honesty tests --check-only --diff

# Run type checking
echo "Running mypy..."
poetry run mypy honesty --ignore-missing-imports

# Test CLI functionality
echo "Testing CLI functionality..."
poetry run python -c "from honesty.cli.main import create_parser; parser = create_parser(); print('CLI parser created successfully')"

echo "All tests completed!" 