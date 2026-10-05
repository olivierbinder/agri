# Application Gradio

<div style="padding: 1rem 1.25rem; border-left: 0.28rem solid #448aff; background: rgba(68, 138, 255, 0.10); border-radius: 0.25rem; font-size: 1.08rem; line-height: 1.5;">
L'application Gradio rend le modèle <strong>manipulable par un utilisateur métier</strong> : choisir un contexte, prédire un rendement, ou comparer toutes les cultures.
</div>

## Parcours utilisateur

[![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://www.gradio.app/)

```mermaid
flowchart LR
    A[Choisir le mode] --> B{Prédire ou<br/>Recommander ?}
    B -->|Prédire| C[Sélectionner la culture]
    B -->|Recommander| D[Toutes les cultures]
    C --> E[Régler zone / année /<br/>pluie / pesticides / température]
    D --> E
    E --> F[Appel API]
    F --> G[Résultat affiché]
```

- **Onglet "🔮 Predict a yield"** : rendement prédit pour la culture choisie, dans le contexte défini au-dessus (zone, année, pluie, pesticides, température, partagé entre les deux onglets).
- **Onglet "🏆 Recommend the best crop"** : classement de toutes les cultures connues par score relatif, sous forme de tableau détaillé.

## Lien avec l'API

| Endpoint | Usage dans l'app |
| --- | --- |
| `/predict` | Calcule le rendement pour la culture sélectionnée |
| `/recommend` | Classe toutes les cultures pour le contexte courant |

`API_URL` est lue depuis la variable d'environnement `API_URL`, injectée par le déploiement Cloud Run (`--set-env-vars`, avec l'URL du service API) et valant `http://localhost:8000` par défaut en local ([src/agri/ui/app.py](../src/agri/ui/app.py)).

## Démo

!!! tip "Démo à ouvrir"
    - **Application déployée** : [agri-ui-28873275232.europe-west1.run.app](https://agri-ui-28873275232.europe-west1.run.app)

    Lancer l'application avec **`just ui`** (ou `just app` pour API + UI ensemble), puis ouvrir :

    - **Application Gradio** : [http://localhost:7860](http://localhost:7860)

??? info "Annexes"

    ## Déploiement

    L'application est un service **Google Cloud Run** (`agri-ui`), déployé par le même job `deploy` que l'API : l'image est construite depuis [Dockerfile.ui](../Dockerfile.ui) — volontairement légère (Gradio, pandas, requests, sans mlflow/xgboost/shap/pandera) —, poussée sur Docker Hub, puis déployée avec `--min-instances 0 --max-instances 1` et l'URL de l'API injectée dans `API_URL`. Détail dans [CI/CD](05_cicd.md).

    ## Cold start

    Avec `--min-instances 0` (indispensable pour ne rien payer au repos), la première visite après une période d'inactivité attend le réveil du conteneur — compter quelques secondes, un peu plus si l'API doit elle aussi démarrer et charger le modèle. C'est le prix du « 0 € ».

    ## Gestion des erreurs

    Si l'API n'est pas joignable à `API_URL` (ex. non déployée), l'app affiche un message d'erreur explicite (`gr.Error`, levé sur `requests.exceptions.RequestException`) plutôt que de planter.
