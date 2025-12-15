#!/bin/bash
set -e

echo "Processing SQL dump to remove DEFINER clauses..."

# Decompress, remove DEFINER clauses, and pipe to MySQL
gunzip -c /tmp/cas-db.sql.gz | \
  sed -E 's/DEFINER[ ]*=[ ]*(`[^`]+`|[^ ]+)@(`[^`]+`|[^ ]+)//g' | \
  mysql -u root -p"${MYSQL_ROOT_PASSWORD}" "${MYSQL_DATABASE}"

echo "Database initialized successfully!"
