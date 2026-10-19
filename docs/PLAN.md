# Plan d'exécution

Une case n'est cochée qu'après exécution réussie de sa commande de validation.

| # | Étape | Validation | Fait |
|---|---|---|---|
| 1 | Squelette, `.gitignore`, `pyproject.toml`, `tasks.ps1` | `git status` n'affiche ni `.env` ni données | [ ] |
| 2 | Environnement Python 3.11 + torch CPU | `.\tasks.ps1 setup` | [ ] |
| 3 | Configuration YAML + modules cœur (vidéo, détection, suivi, équipes, terrain, statistiques, rendu) | `.\tasks.ps1 lint` | [ ] |
| 4 | Tests synthétiques sans réseau | `.\tasks.ps1 test` | [ ] |
| 5 | Datasets Roboflow + entraînements CPU plafonnés (arrière-plan) | `.\tasks.ps1 train` → `models/*.pt` | [ ] |
| 6 | Pipeline complet : vidéo annotée, CSV, heatmaps, JSON | `.\tasks.ps1 process` | [ ] |
| 7 | Contrôle visuel d'une image annotée | image extraite dans `logs/` | [ ] |
| 8 | Application Streamlit | `streamlit run` démarré puis arrêté | [ ] |
| 9 | Outils d'évaluation (IDF1/MOTA, équipes, FPS) | `python -m arabcupvision.cli benchmark` | [ ] |
| 10 | Docker | `docker compose build` (ou relecture si Docker indisponible) | [ ] |
| 11 | README, METHODOLOGY | relecture, aucun chiffre de performance | [ ] |
| 12 | Hygiène : secrets, taille des fichiers, historique | `detect-secrets scan`, `git log --format=%B` | [ ] |
| 13 | Push final | `git push origin main`, `git status` propre | [ ] |
