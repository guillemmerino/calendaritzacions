#!/usr/bin/env bash
set -e

# Activa el venv
. /venv/bin/activate

# Si hi ha requirements, instal·la només si han canviat
if [ -f /app/requirements.txt ]; then
  if [ ! -f /venv/.req.hash ] || ! cmp -s /app/requirements.txt /venv/.req.hash; then
    echo "🔧 Instal·lant dependències (canvi detectat a requirements.txt)..."
    pip install --upgrade pip
    pip install -r /app/requirements.txt
    cp /app/requirements.txt /venv/.req.hash
  else
    echo "✅ Dependències al dia (sense canvis a requirements.txt)."
  fi
else
  echo "ℹ️ No s'ha trobat requirements.txt"
fi

exec "$@"
