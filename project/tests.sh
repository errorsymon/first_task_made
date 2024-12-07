#!/bin/bash
# This is the tests.sh shell script to run pytest

# Ensure that you're in the right directory (optional)
# cd /path/to/your/tests

# Run pytest and point it to the tests.py file (if necessary, use the path to tests.py)
pytest --maxfail=1 --disable-warnings -q ./project/tests.py

