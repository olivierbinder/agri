# Contexte du projet

<div style="padding: 1rem 1.25rem; border-left: 0.28rem solid #448aff; background: rgba(68, 138, 255, 0.10); border-radius: 0.25rem; font-size: 1.08rem; line-height: 1.5;">
Prédire et recommander des <strong>rendements agricoles</strong> à partir de données climatiques et agricoles (pluviométrie, pesticides, température), à partir du dataset FAO (Food and Agriculture Organization).
</div>

## Schéma général

[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-006ACC?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/features/actions)

```mermaid
flowchart LR
    A[Données FAO<br/>yield_df.csv] --> B[Préparation +<br/>feature engineering]
    B --> C[Modèle XGBoost<br/>tuné via GridSearchCV]
    C --> D[MLflow Registry<br/>alias Champion]
    D --> E[API FastAPI]
    E --> F[App Streamlit]
    D -.export bundle.-> G[Image Docker]
```

L'API et le frontend sont déployés indépendamment :

- l'**API** est packagée dans une image Docker (le modèle `Champion` y est embarqué), construite et poussée sur Docker Hub par la CI/CD ;
- l'**application Streamlit** est déployée séparément sur Streamlit Community Cloud, connectée directement à ce dépôt GitHub.

## Stack MLOps

| Brique | Rôle |
| --- | --- |
| `pandera` | Validation des schémas de données (`InputsSchema`, `TargetsSchema`, `OutputsSchema`) |
| `scikit-learn` / `XGBoost` | Preprocessing (`TargetEncoder`) et modèle de régression |
| `SHAP` | Explicabilité du modèle (importances, valeurs SHAP) |
| `MLflow` | Tracking des expérimentations, model registry, alias `Champion` |
| `FastAPI` / `Pydantic` | Service HTTP de prédiction, validation des requêtes |
| `Streamlit` | Interface utilisateur métier |
| `Docker` / `Docker Hub` | Packaging et distribution de l'image de l'API |
| `GitHub Actions` | Tests, build et publication automatisés |
| `uv` / `just` | Gestion d'environnement et raccourcis de commandes |

## Pages de cette documentation

- **[Modèle](02_model.md)** — préparation des données, feature engineering, entraînement et tuning.
- **[API](03_api.md)** — le service FastAPI qui sert le modèle.
- **[Application](04_prediction.md)** — l'interface Streamlit.
- **[CI/CD](05_cicd.md)** — le pipeline de tests et de build.
- **[Dépôt](06_depot.md)** — structure du code et synthèse.
- **[Architecture du code](07_architecture.md)** — `core`, `io`, `jobs`, `utils`, `confs` : le cœur data science, piloté par MLflow.
