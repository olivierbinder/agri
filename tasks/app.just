# Démarrer l'API (Backend)
api port="8000":
    uv run uvicorn agri.api.server:app --port {{port}} --reload

# Démarrer Streamlit (Frontend)
ui port="8501":
    uv run streamlit run src/agri/ui/app.py --server.port {{port}}

# Démarrer FastAPI et Streamlit simultanément
app:
    @echo "🚀 Démarrage de FastAPI (port 8000) et Streamlit (port 8501)..."
    uv run uvicorn agri.api.server:app --port 8000 & \
    uv run streamlit run src/agri/ui/app.py --server.port 8501 & \
    wait
