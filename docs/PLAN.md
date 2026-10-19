# Plan d'exécution

Une case n'est cochée qu'après exécution réussie de sa commande de validation.

| # | Étape | Validation | Fait |
|---|---|---|---|
| 1 | Squelette, `.gitignore`, `pyproject.toml`, `tasks.ps1` | `git status` n'affiche ni `.env` ni données | [x] |
| 2 | Environnement Python 3.11 + torch CPU | `.\tasks.ps1 setup` | [x] |
| 3 | Configuration YAML + modules cœur (vidéo, détection, suivi, équipes, terrain, statistiques, rendu) | `.\tasks.ps1 lint` | [x] |
| 4 | Tests synthétiques sans réseau | `.\tasks.ps1 test` | [x] |
| 5 | Datasets Roboflow + entraînements CPU plafonnés | datasets téléchargés, entraînement démarré puis interrompu à la demande (voir DECISIONS) ; repli COCO | [~] |
| 6 | Pipeline complet : vidéo annotée, CSV, heatmaps, JSON | `.\tasks.ps1 process` (vidéo synthétique) | [x] |
| 7 | Contrôle visuel d'une image annotée | image extraite de `annotated.mp4` (ellipses, équipes, arbitre, mini-carte) | [x] |
| 8 | Application Streamlit | `streamlit run` (health ok) + `AppTest` (chargement et résultats) | [x] |
| 9 | Outils d'évaluation (IDF1/MOTA, équipes, FPS) | `eval-tracking`, `eval-teams`, `benchmark` exécutés sur la vidéo synthétique | [x] |
| 10 | Docker | `docker compose build` + conteneur démarré (health ok) | [x] |
| 11 | README, METHODOLOGY | relecture, aucun chiffre de performance | [x] |
| 12 | Hygiène : secrets, taille des fichiers, historique | `detect-secrets scan`, `git log --format=%B` | [x] |
| 13 | Push final | `git push origin main`, `git status` propre | [x] |
