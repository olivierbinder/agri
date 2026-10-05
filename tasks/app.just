# Démarrer l'API (Backend)
api port="8000":
    uv run uvicorn agri.api.server:app --port {{port}} --reload

# Démarrer Gradio (Frontend)
ui port="7860" api_port="8000":
    GRADIO_SERVER_PORT={{port}} API_PORT={{api_port}} uv run python src/agri/ui/app.py

# Démarrer FastAPI et Gradio simultanément, avec arrêt propre
app api_port="8000" ui_port="7860":
    #!/usr/bin/env bash
    set -uo pipefail

    # Tue tout le groupe de process à la sortie / Ctrl+C
    trap 'echo "Arrêt des serveurs..."; kill 0 2>/dev/null' INT TERM EXIT

    echo "Démarrage de FastAPI (port {{api_port}}) et Gradio (port {{ui_port}})..."
    uv run uvicorn agri.api.server:app --port {{api_port}} &
    api_pid=$!
    GRADIO_SERVER_PORT={{ui_port}} API_PORT={{api_port}} uv run python src/agri/ui/app.py &
    ui_pid=$!

    # macOS ships bash 3.2 (no `wait -n`): poll instead, and stop as soon as one dies
    while kill -0 "$api_pid" 2>/dev/null && kill -0 "$ui_pid" 2>/dev/null; do
        sleep 1
    done

    if kill -0 "$api_pid" 2>/dev/null; then
        echo "Gradio s'est arrêté."
    else
        echo "FastAPI s'est arrêté."
    fi
