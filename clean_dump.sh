#!/bin/bash
set -e

echo "Cleaning SQL dump (this is a one-time process)..."
echo "Input: ./data/CASBotanybackup2025-01-23.sql.gz"
echo "Output: ./data/CASBotanybackup2025-01-23-cleaned.sql.gz"

# Decompress, remove DEFINER clauses, and recompress
gunzip -c ./data/CASBotanybackup2025-01-23.sql.gz | \
  sed -E 's/DEFINER[ ]*=[ ]*(`[^`]+`|[^ ]+)@(`[^`]+`|[^ ]+)//g' | \
  gzip > ./data/CASBotanybackup2025-01-23-cleaned.sql.gz

echo "✓ Cleaned dump created!"
echo "Update your .env file to use: SQL_DUMP_PATH=./data/CASBotanybackup2025-01-23-cleaned.sql.gz"
