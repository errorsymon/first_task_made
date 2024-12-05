#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define output directory and expected files
OUTPUT_DIR="$(pwd)/test_outputs"
FINAL_DATA_FILE="${OUTPUT_DIR}/Final_Data.csv"
INDICATOR_DATA_FILE="${OUTPUT_DIR}/Indicator_data.csv"
IMPORTANT_FEATURES_FILE="${OUTPUT_DIR}/Important_Features.csv"

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

# Run the pipeline
echo "Running the data pipeline..."
python pipline.py  # Use 'python' for better Windows compatibility

# Check if the expected output files are created
echo "Checking if output files are generated..."

if [[ -f "$FINAL_DATA_FILE" ]]; then
    echo "Final data file exists: $FINAL_DATA_FILE"
else
    echo "Error: Final data file does not exist!" >&2
    exit 1
fi

if [[ -f "$INDICATOR_DATA_FILE" ]]; then
    echo "Indicator data file exists: $INDICATOR_DATA_FILE"
else
    echo "Error: Indicator data file does not exist!" >&2
    exit 1
fi

if [[ -f "$IMPORTANT_FEATURES_FILE" ]]; then
    echo "Important features file exists: $IMPORTANT_FEATURES_FILE"
else
    echo "Error: Important features file does not exist!" >&2
    exit 1
fi

# Perform basic structure checks
echo "Performing basic checks on the final data file..."
head -n 5 "$FINAL_DATA_FILE" || {
    echo "Error: Unable to read $FINAL_DATA_FILE" >&2
    exit 1
}

echo "Performing basic checks on the indicator data file..."
head -n 5 "$INDICATOR_DATA_FILE" || {
    echo "Error: Unable to read $INDICATOR_DATA_FILE" >&2
    exit 1
}

echo "Performing basic checks on the important features file..."
head -n 5 "$IMPORTANT_FEATURES_FILE" || {
    echo "Error: Unable to read $IMPORTANT_FEATURES_FILE" >&2
    exit 1
}

# Print success message
echo "All tests passed successfully!"
