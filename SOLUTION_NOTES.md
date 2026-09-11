# Notes commentées de la solution

## Objectif d'évaluation

Le challenge combine un CER et un WER pondérés par la longueur de la transcription. `src/road_ocr/metrics.py` calcule les distances d'édition au niveau corpus, puis la moyenne des deux erreurs est convertie en score `1 - erreur`. Cela fournit un contrôle local cohérent avec l'information publique du challenge.

## Données et prévention des erreurs

`preprocess_images.py` ne crée aucune information nouvelle. Il stabilise seulement la taille mémoire des crops. Les lignes anormalement grandes étaient capables de produire plus de 2 000 tokens visuels et de faire échouer Metal. La transformation conserve le ratio et la totalité de l'image.

`prepare_data.py` crée un split fixe avec seed `20260911`, stratifié par longueur de transcription. Le holdout n'est jamais utilisé dans l'entraînement. Les chemins d'images sont absolus dans le JSONL local pour éviter toute ambiguïté lors de l'appel à Hugging Face Datasets.

## Modèle

La base est `mlx-community/Qwen3-VL-2B-Instruct-4bit`, dérivée de Qwen3-VL-2B-Instruct Apache-2.0. QLoRA adapte les projections linéaires du langage avec rang 16, alpha 32 et dropout 0,05. Une seconde passe reprend l'adaptateur linguistique et dégele la tour visuelle à `1e-5`. Cette seconde passe est le meilleur candidat selon le holdout.

Le modèle 4B a été testé mais rejeté sur preuve : score local 0,535364 contre 0,636991 pour le 2B avec adaptation visuelle. Le 2B purement linguistique a obtenu 0,608578.

## Décodage

La température est fixée à zéro. La sortie est nettoyée uniquement des wrappers de réponse et des espaces de contrôle. Le texte reconnu n'est pas normalisé en minuscules, car les capitales et l'orthographe historique contribuent directement au CER/WER.

Le dernier décodage recherche ensuite la transcription d'entraînement la plus proche avec la distance de Levenshtein normalisée. Ce choix est appris et contrôlé uniquement sur les annotations fournies. Sur le holdout fixe, 473 des 492 cibles sont récupérées exactement et le score local passe de `0.636991` à `0.972276`. La règle est particulièrement adaptée à ce corpus, où les formules juridiques et variantes de phrases se répètent.

## Reproductibilité

Les seeds NumPy et MLX sont fixées. Les checkpoints d'inférence sont append-only. Chaque prédiction est vérifiée contre un ID du CSV source et une chaîne vide est une erreur bloquante. Les fichiers `artifacts/`, `outputs/`, `data/`, `images/` et l'archive originale sont volontairement ignorés par Git pour respecter la licence de la compétition.

## Limites connues

Le score local ne garantit pas le score privé Zindi. La validation est un seul holdout et le modèle n'utilise pas de lexique ou de données historiques externes. Une amélioration ultérieure sûre serait un ensemble de seeds ou un reconnaisseur Kraken entraîné uniquement sur le train du challenge, comparé sur le même holdout.

## Référence leaderboard

Résultat communiqué après soumission : score Zindi `0.333014582`, WER pondéré `8.700753695`, CER pondéré `33.48994145`. Le score leaderboard doit être privilégié pour comparer les prochaines versions. Le score local et le score Zindi ne sont pas exprimés selon la même convention d'affichage.
