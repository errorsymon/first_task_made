#!/bin/bash
set -e

# Ensure that pytest is installed
if ! command -v pytest &> /dev/null; then
  echo "Error: pytest is not installed. Please ensure pytest is installed in the environment."
  exit 1
fi

pytest --maxfail=1 --disable-warnings -q ./project/tests.py

# Optional: Print a success message
echo "Tests completed successfully."
