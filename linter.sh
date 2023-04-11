#!/bin/bash
echo "Running isort..."
isort .

echo "Running black..."
black .

echo "Running flake..."
flake8 .
