# Méthodologie

Ce document décrit le fonctionnement de chaque brique du pipeline. Les paramètres cités sont ceux de `configs/default.yaml`.

## 1. Lecture vidéo

`video.iter_frames` lit la vidéo avec OpenCV, saute `frame_skip - 1` images sur `frame_skip` (avec `grab`, sans décodage complet), s'arrête après `max_seconds` et redimensionne à `resize_width` (dimensions paires). L'indice d'origine de chaque image est conservé : les temps (`time_s`) et l'évaluation restent alignés sur la vidéo source.

## 2. Détection

- **Modèle fine-tuné** (`models/detector.pt`) : YOLOv8n entraîné sur *Football Players Detection* (classes `ball`, `goalkeeper`, `player`, `referee`).
- **Repli COCO** : si le fichier est absent, YOLOv8n COCO est utilisé et seules les classes `person` et `sports ball` sont conservées. Tout modèle exposant une classe `person` est traité de la même façon.
- Les noms de classes sont ramenés à quatre classes canoniques (`detection.canonical_class`), ce qui rend le pipeline indépendant de l'ordre des classes du modèle.
- Le ballon a son propre seuil de confiance (`ball_confidence`) ; seule la détection la plus confiante par image est gardée.

## 3. Suivi

- Les boîtes de personnes sont transmises au tracker ByteTrack (ou BoT-SORT, `tracking.tracker`) d'ultralytics, appelé directement plutôt que via `model.track`, afin de traiter le ballon séparément. Le tampon des pistes perdues est adapté au débit effectif (après saut d'images).
- Les pistes de moins de `min_track_length` observations sont supprimées (fausses détections).
- Les positions sur le terrain sont lissées par piste avec une moyenne glissante centrée (`smoothing_window`).
- Ballon : les positions manquantes sont interpolées linéairement, uniquement à l'intérieur de la trajectoire (pas d'extrapolation) et seulement pour les trous d'au plus `ball_max_gap` images traitées. Les positions interpolées sont signalées (`interpolated`) et dessinées en contour.

## 4. Équipes

1. **Découpe** : bande verticale `[crop_top, crop_bottom]` de la boîte, rognée de `crop_side` de chaque côté — le haut du maillot, en évitant la tête, le short et le fond.
2. **Masquage de la pelouse** : en HSV, les pixels de teinte comprise dans `green_hue_range` et de saturation supérieure à `green_min_saturation` sont écartés.
3. **Couleur** : médiane Lab des pixels restants (Lab rend les distances plus proches de la perception que RGB). En dessous de `min_pixels`, aucune couleur n'est retenue.
4. **K-Means à deux classes** ajusté sur les couleurs des joueurs des `fit_frames` premières images. Les centres sont remplacés par la médiane de chaque cluster et un seuil par cluster (`outlier_factor` × distance médiane, au moins `min_outlier_distance`) identifie les couleurs atypiques. Le K-Means est ensuite ré-ajusté sans elles : un arbitre ou un gardien ne peut plus « tirer » un centre.
5. **Vote majoritaire** : chaque piste de joueur reçoit l'équipe majoritaire de ses observations (égalité : plus petite étiquette).
6. **Gardiens** : rattachés à l'équipe dont le centre de gravité (position moyenne des joueurs sur le terrain) est le plus proche de leur position moyenne. **Arbitres** : exclus des équipes et des statistiques.
7. **Repli COCO** : sans classes gardien/arbitre, une piste majoritairement atypique devient un gardien si sa position moyenne est derrière tous les joueurs de champ, sinon un arbitre.

## 5. Terrain et homographie

- Le terrain réglementaire mesure 105 × 68 m, origine au coin supérieur gauche (`pitch.geometry.PitchDimensions`).
- Le modèle de points clés (YOLOv8n-pose, *Football Field Detection*) prédit 32 points. Leur position réglementaire (`pitch.keypoints_m`) a été établie en dessinant les indices annotés sur des images du dataset ; elle est cohérente avec la symétrie gauche/droite (`flip_idx`) du dataset :

| Indices | Points |
|---|---|
| 0–5 | ligne de but gauche : coin haut, surface de réparation, surface de but (×2), surface de réparation, coin bas |
| 6–8 | coins de la surface de but gauche, point de penalty gauche |
| 9–12 | ligne de la surface de réparation gauche : coin, intersections avec l'arc (×2), coin |
| 13–16 | ligne médiane : touche haute, rond central haut et bas, touche basse |
| 17–23 | symétriques à droite de 9–12 et 6–8 |
| 24–29 | ligne de but droite |
| 30–31 | rond central gauche et droit |

- Les points de confiance ≥ `keypoint_confidence` sont utilisés si leur nombre atteint `min_keypoints` ; l'homographie est estimée par RANSAC (`ransac_threshold_m`) puis rejetée si l'erreur médiane de reprojection est trop grande.
- Le recalcul a lieu toutes les `recompute_interval` images ; entre deux recalculs, ou si la confiance est insuffisante, la **dernière homographie valide** est conservée.
- Sans modèle : calibrage manuel (`<vidéo>.calibration.json` ou `--calibration`), sinon homographie approximative supposant que l'image couvre tout le terrain. La source utilisée pour chaque image est comptée dans `summary.json` (`homography_sources`).
- Chaque joueur est projeté par son **point au sol** (milieu du bord inférieur de la boîte).

## 6. Statistiques

Pour chaque piste, triée par temps :

- pas `i` : déplacement `d_i` (m) et durée `dt_i` (s) ; vitesse `v_i = d_i / dt_i × 3,6` (km/h) ;
- un pas avec `v_i > max_speed_kmh` est considéré comme une erreur (saut d'identité, homographie instable) : il n'entre ni dans la distance, ni dans les vitesses ;
- distance = somme des pas valides ; vitesse instantanée = moyenne glissante centrée (`speed_window`) des vitesses valides ; vitesse maximale = maximum de la vitesse lissée ;
- temps de présence = nombre d'observations × durée représentée par une image traitée (`frame_skip / fps`).

## 7. Heatmaps

Histogramme 2D des positions sur une grille `heatmap_bins` couvrant le terrain, lissé par un filtre gaussien (`heatmap_sigma`), normalisé à 1, puis superposé (matplotlib) à un terrain dessiné avec OpenCV. Une heatmap est produite par équipe et par joueur suivi au moins `heatmap_min_frames` images.

## 8. Possession

À chaque image où le ballon est connu (détecté ou interpolé), le joueur le plus proche sur le terrain prend la possession s'il est à moins de `possession_max_distance_m` ; sinon l'image n'est attribuée à personne. La part de chaque équipe est calculée sur les images attribuées. La vidéo affiche la possession cumulée.

## 9. Rendu

Seconde passe sur la vidéo : ellipse partielle sous les pieds (couleur de l'équipe, jaune pour l'arbitre, gris si inconnue), numéro de piste, triangle au-dessus du ballon, bandeau de temps et de possession, et mini-carte tactique (terrain vu de dessus, positions lissées) incrustée en bas de l'image. La vidéo est écrite en MPEG-4 puis ré-encodée en H.264 avec le ffmpeg d'`imageio-ffmpeg` pour être lisible dans un navigateur.

## 10. Vidéo synthétique

`arabcupvision.cli synth` dessine un terrain en perspective (homographie connue), deux équipes de huit joueurs, deux gardiens, un arbitre et un ballon animés. Elle est fournie avec son calibrage et sa vérité terrain MOTChallenge. Elle sert uniquement à tester le pipeline de bout en bout ; elle ne remplace pas une vidéo de match pour juger de la qualité des modèles.

## 11. Reproductibilité

Graine unique (`seed`) appliquée à Python, NumPy, PyTorch, au K-Means et aux entraînements (`deterministic=True`). Toute la configuration est dans un seul fichier YAML validé par pydantic.
