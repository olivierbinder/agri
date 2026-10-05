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

# run the API (Docker) and the UI (local) together, like the deployed setup
[group('docker')]
docker-app: docker-build
    docker run --rm -d -p 8000:8000 --name agri-api-tmp agri-api
    API_URL=http://localhost:8000 uv run python src/agri/ui/app.py || true
    docker stop agri-api-tmp

# build the UI's Docker image (lightweight, Gradio only)
[group('docker')]
docker-build-ui:
    docker build -f Dockerfile.ui -t agri-ui .

# run the UI's Docker image locally (:7860), pointing at an API on the host's :8000
# (start the API first with `just docker-run` or `just api`)
[group('docker')]
docker-run-ui:
    docker run --rm -p 7860:7860 -e API_URL=http://host.docker.internal:8000 agri-ui
