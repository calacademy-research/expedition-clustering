#!/bin/bash
set -e

echo "Processing SQL dump with optimizations..."

# Disable safety checks for faster import, remove DEFINER clauses, then restore safety
gunzip -c /tmp/cas-db.sql.gz | \
  sed -E 's/DEFINER[ ]*=[ ]*(`[^`]+`|[^ ]+)@(`[^`]+`|[^ ]+)//g' | \
  mysql -u root -p"${MYSQL_ROOT_PASSWORD}" "${MYSQL_DATABASE}" \
    --init-command="SET autocommit=0; SET unique_checks=0; SET foreign_key_checks=0;"

# Re-enable safety checks
mysql -u root -p"${MYSQL_ROOT_PASSWORD}" "${MYSQL_DATABASE}" \
  -e "SET autocommit=1; SET unique_checks=1; SET foreign_key_checks=1;"

echo "Database initialized successfully!"
