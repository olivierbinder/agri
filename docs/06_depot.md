# Structure du dépôt et synthèse

<div style="padding: 1rem 1.25rem; border-left: 0.28rem solid #448aff; background: rgba(68, 138, 255, 0.10); border-radius: 0.25rem; font-size: 1.08rem; line-height: 1.5;">
Le dépôt suit les étapes du projet : <strong>données → modèle → API → interface → CI/CD</strong>.
</div>

## Vue rapide

### Données et modèle

```text
agri/
|-- data/
|   |-- crop_yield.csv        # dataset FAO brut
|   `-- processed/            # inputs/targets train/test
|-- confs/                    # config YAML de chaque job (Pydantic + discriminants KIND)
|   |-- tuning.yaml
|   |-- training.yaml
|   |-- evaluations.yaml
|   |-- explanations.yaml
|   |-- promotion.yaml
|   `-- inference.yaml
`-- mlruns.db / mlruns/       # tracking MLflow local
```

### Code applicatif

```text
src/agri/
|-- core/          # features, models, metrics, schemas, constants
|-- io/            # datasets, registries (saver/loader/register), services (MlflowService)
|-- jobs/          # TuningJob, TrainingJob, EvaluationsJob, ExplanationsJob, PromotionJob, InferenceJob
|-- utils/         # splitters, searchers, signers
|-- api/           # server (FastAPI), logic, dependencies
`-- ui/            # app.py (Gradio)
```

### Exploitation

```text
agri/
|-- tests/                    # miroir de src/agri, dont tests/api/
|-- deploy/model/             # modèle Champion bundlé dans l'image de l'API
|-- deploy/reference/         # rendements réels 2013 ("actual vs predicted")
|-- .github/workflows/        # ci-cd.yml (tests, build, déploiement), docs.yml
|-- Dockerfile                # image API : FastAPI + modèle
|-- Dockerfile.ui             # image UI : Gradio seul, sans stack ML
|-- requirements-ui.txt       # dépendances de l'image UI
|-- docker-compose.yml
`-- docs/                     # cette documentation (Zensical)
```

## Schéma final

```mermaid
flowchart LR
    subgraph DATA["<b>1. Données</b>"]
        direction TB
        RAW["data/crop_yield.csv"]
        SPLIT["data/split_data.py"]
        PROC["data/processed/*.csv"]
        RAW --> SPLIT --> PROC
    end

    subgraph TRAIN["<b>2. Entraînement et registre</b>"]
        direction TB
        TUNE["TuningJob<br/>GridSearchCV"]
        TRAINJOB["TrainingJob"]
        MLFLOW["MLflow Registry<br/>versions"]
        CHAMP["Alias Champion<br/>(PromotionJob)"]
        TUNE --> TRAINJOB --> MLFLOW --> CHAMP
    end

    subgraph API["<b>3. API de prédiction</b>"]
        direction TB
        SCHEMAS["schemas.py<br/>Pydantic + Pandera"]
        LOGIC["logic.py<br/>predict / recommend"]
        FASTAPI["server.py<br/>FastAPI routes"]
        SCHEMAS --> FASTAPI
        LOGIC --> FASTAPI
    end

    subgraph UI["<b>4. Interface Gradio</b>"]
        direction TB
        APP["ui/app.py"]
    end

    subgraph OPS["<b>5. Qualité, publication et déploiement</b>"]
        direction TB
        TESTS["tests/"]
        CI["GitHub Actions<br/>ruff + ty + pytest"]
        DOCKER["Dockerfile<br/>API + modèle bundlé"]
        DOCKERUI["Dockerfile.ui<br/>UI Gradio seule"]
        HUB["Docker Hub<br/>agri-api + agri-ui"]
        RUN["Cloud Run<br/>agri-api + agri-ui"]
        TESTS --> CI --> DOCKER --> HUB --> RUN
        CI --> DOCKERUI --> HUB
    end

    PROC --> TUNE
    PROC --> TRAINJOB
    CHAMP --> LOGIC
    CHAMP --> DOCKER
    FASTAPI --> APP
    FASTAPI --> DOCKER
    APP --> DOCKERUI

    classDef data fill:#E8F5E9,stroke:#2E7D32,color:#1B5E20
    classDef train fill:#FFF3E0,stroke:#EF6C00,color:#4E342E
    classDef api fill:#E0F2F1,stroke:#00897B,color:#004D40
    classDef ui fill:#FCE4EC,stroke:#C2185B,color:#880E4F
    classDef ops fill:#EDE7F6,stroke:#5E35B1,color:#311B92

    class RAW,SPLIT,PROC data
    class TUNE,TRAINJOB,MLFLOW,CHAMP train
    class SCHEMAS,LOGIC,FASTAPI api
    class APP ui
    class TESTS,CI,DOCKER,DOCKERUI,HUB,RUN ops
```


??? info "Annexes"

    ## Fichiers racine

    - `pyproject.toml` : dépendances du runtime API + extras `ui` / `eda` / `explain`, et groupes `check` / `commit` / `dev` / `docs` / `mlflow` / `notebook`.
    - `uv.lock` : versions figées pour reproduire l'environnement.
    - `justfile` + `tasks/*.just` : raccourcis `api`, `ui`, `app`, `check`, `docker-*`, `docs-*`, `mlflow-*`.
    - `zensical.toml` : navigation et configuration de cette documentation.
    - `MLproject` + `python_env.yaml` : point d'entrée MLflow Projects pour rejouer un job (`mlflow run . -P conf_file=confs/training.yaml`).

    ## Artefacts de production

    - `deploy/model/` : version `Champion` du registre MLflow, bundlée dans l'image Docker (`just docker-export-model`).
    - `deploy/reference/` : rendements réels 2013, bundlés eux aussi pour la comparaison « prédit vs réel » de l'API.
    - `data/processed/*.csv` : jeux d'entraînement/test après feature engineering.
    - `outputs/*.csv` : prédictions et explications générées par `InferenceJob` / `ExplanationsJob`.
