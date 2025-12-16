#!/bin/bash
# Clean SQL dump by removing DEFINER clauses for MySQL 5.7+ compatibility
# Usage: ./scripts/clean_dump.sh <input.sql.gz> <output.sql.gz>

set -e

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input.sql.gz> <output.sql.gz>"
    echo "Example: $0 ./data/raw-dump.sql.gz ./data/cleaned-dump.sql.gz"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

echo "Cleaning SQL dump..."
echo "  Input:  $INPUT_FILE"
echo "  Output: $OUTPUT_FILE"

# Decompress, remove DEFINER clauses, and recompress
gunzip -c "$INPUT_FILE" | \
  sed -E 's/DEFINER[ ]*=[ ]*(`[^`]+`|[^ ]+)@(`[^`]+`|[^ ]+)//g' | \
  gzip > "$OUTPUT_FILE"

echo "✓ Cleaned dump created successfully!"
