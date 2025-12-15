#!/usr/bin/env bash
# Clean a MySQL dump for MySQL 5.7/8+: remove NO_AUTO_CREATE_USER and rewrite definers to CURRENT_USER.
# Usage: scripts/clean_dump.sh path/to/source.sql.gz path/to/output.sql.gz

set -euo pipefail

src="${1:-}"
dst="${2:-}"

if [[ -z "${src}" || -z "${dst}" ]]; then
  echo "Usage: $0 source.sql.gz output.sql.gz" >&2
  exit 1
fi

if [[ ! -f "${src}" ]]; then
  echo "Source dump not found: ${src}" >&2
  exit 1
fi

mkdir -p "$(dirname "${dst}")"

gzip -cd "${src}" \
  | perl -pe 's/NO_AUTO_CREATE_USER//gi; s/DEFINER=`[^`]+`@`[^`]+`/DEFINER=CURRENT_USER/gi' \
  | gzip > "${dst}"

echo "Wrote cleaned dump to ${dst}"
