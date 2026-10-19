# Journal des décisions

| Date | Décision | Raison |
|---|---|---|
| 2026-10-19 | Aucune vidéo dans `data/videos/` : une vidéo synthétique (terrain en perspective, joueurs dessinés) est générée par `arabcupvision.cli synth`, uniquement pour les tests et la démonstration du rendu. | Pas de vidéo de match fournie ; pas de redistribution d'images protégées. |
| 2026-10-19 | `opencv-python-headless` au lieu de `opencv-python`. | Pas besoin de fenêtres GUI ; image Docker plus légère, pas de dépendance X11. |
| 2026-10-19 | Écriture vidéo en `mp4v` puis ré-encodage H.264 via le binaire ffmpeg fourni par `imageio-ffmpeg` quand il est disponible. | Les navigateurs (Streamlit) ne lisent pas le MPEG-4 Part 2. |
| 2026-10-19 | Traitement en deux passes : (1) détection, suivi, couleurs, homographie ; (2) rendu avec les équipes et trajectoires finales. | Le vote majoritaire et le lissage nécessitent la piste complète avant d'annoter. |
| 2026-10-19 | Calibrage manuel optionnel (fichier JSON de 4+ correspondances image → terrain) utilisé si le modèle de points clés est absent. | Permet des statistiques métriques sans modèle de points clés (et sur la vidéo synthétique). |
| 2026-10-19 | Dernier recours sans points clés ni calibrage : homographie approximative supposant que l'image couvre tout le terrain, signalée `approx` dans le JSON récapitulatif. | Le pipeline produit toujours ses sorties ; la source de l'homographie reste traçable. |
| 2026-10-19 | `lap` ajouté aux dépendances. | Requis par le tracker ByteTrack d'ultralytics. |
| 2026-10-19 | Correspondance des 32 points clés du dataset `football-field-detection` avec le terrain établie en dessinant les indices annotés sur une image du dataset (cohérente avec son `flip_idx`) ; stockée dans `configs/default.yaml`. | Le dataset ne fournit que des indices ; la géométrie est déduite des marquages réglementaires. |
| 2026-10-19 | K-Means d'équipes avec centres robustes (médiane par cluster) et ré-ajustement après rejet des couleurs atypiques. | Les couleurs minoritaires (arbitre, gardiens) tiraient les centres moyens et faisaient classer une équipe entière comme atypique. |
| 2026-10-19 | Repli COCO : une piste de couleur atypique est un gardien si sa position moyenne est derrière tous les joueurs de champ, sinon un arbitre. | Le modèle COCO ne distingue que `person` ; règle simple, indépendante du cadrage. |
| 2026-10-19 | Détection et suivi séparés : `predict` puis tracker ByteTrack/BoT-SORT d'ultralytics appelé directement sur les personnes ; le ballon n'est pas suivi (meilleure détection par image). | `model.track` ne renvoie que les pistes confirmées et perdrait le ballon. |
| 2026-10-19 | Entraînements CPU plafonnés : YOLOv8n et YOLOv8n-pose, imgsz 640, ≤ 12/15 epochs et ≤ 0,9 h chacun (`training/train.yaml`). | Pas de GPU ; un notebook Colab permet de réentraîner plus longtemps. |
| 2026-10-19 | Matrice de distances IoU réimplémentée pour motmetrics. | `motmetrics.distances.iou_matrix` appelle `np.asfarray`, supprimé dans NumPy 2. |
| 2026-10-19 | Évaluation de l'application Streamlit par `streamlit.testing.AppTest` (chargement + affichage des résultats d'un traitement) en plus d'un démarrage réel de `streamlit run`. | Vérifie aussi la partie résultats, inaccessible sans import manuel de vidéo. |
