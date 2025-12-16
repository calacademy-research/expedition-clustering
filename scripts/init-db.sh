#!/bin/bash
# Docker entrypoint script to initialize MySQL database with DEFINER clause removal
# This script is automatically run by MySQL container on first startup

set -e

echo "Processing SQL dump to remove DEFINER clauses..."

# Decompress, remove DEFINER clauses, and pipe to MySQL with optimizations
gunzip -c /tmp/cas-db.sql.gz | \
  sed -E 's/DEFINER[ ]*=[ ]*(`[^`]+`|[^ ]+)@(`[^`]+`|[^ ]+)//g' | \
  mysql -u root -p"${MYSQL_ROOT_PASSWORD}" "${MYSQL_DATABASE}" \
    --init-command="SET autocommit=0; SET unique_checks=0; SET foreign_key_checks=0;"

# Re-enable safety checks
mysql -u root -p"${MYSQL_ROOT_PASSWORD}" "${MYSQL_DATABASE}" \
  -e "SET autocommit=1; SET unique_checks=1; SET foreign_key_checks=1;"

echo "Database initialized successfully!"
