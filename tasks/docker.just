# export the local Champion model to deploy/model/, for bundling into the Docker image
[group('docker')]
docker-export-model:
    #!/usr/bin/env bash
    set -euo pipefail
    uv run python3 <<'PY'
    import pathlib
    import shutil

    import mlflow

    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_registry_uri("sqlite:///mlruns.db")

    local_path = mlflow.artifacts.download_artifacts(artifact_uri="models:/agri@Champion")

    dest = pathlib.Path("deploy/model")
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(local_path, dest)

    print(f"Exported Champion model to {dest}/")
    PY

# build the API's Docker image
[group('docker')]
docker-build:
    docker build -t agri-api .

# run the API's Docker image locally (:8000)
[group('docker')]
docker-run:
    docker run --rm -p 8000:8000 agri-api

# run the API (Docker) and Streamlit (local) together, like the deployed setup
[group('docker')]
docker-app: docker-build
    docker run --rm -d -p 8000:8000 --name agri-api-tmp agri-api
    API_URL=http://localhost:8000 uv run streamlit run src/agri/ui/app.py || true
    docker stop agri-api-tmp
