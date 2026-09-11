# R.O.A.D. Barbados Historic Handwriting OCR

Solution reproductible pour le challenge Zindi R.O.A.D. Barbados Historic Handwriting.

Le meilleur modèle local validé est `Qwen3-VL-2B-Instruct-4bit` adapté avec QLoRA puis avec la tour visuelle dégelée. La validation fixe obtient `CER=0.254600`, `WER=0.471418`, soit un score local `0.636991` avec la moyenne des deux métriques. Le score local n'est pas un score leaderboard.

## Reproduire

Les fichiers du challenge doivent être placés localement dans ce dossier, conformément aux règles Zindi. Ils ne sont pas inclus dans le dépôt.

```bash
uv sync
uv run python scripts/preprocess_images.py
PYTHONPATH=src uv run python scripts/prepare_data.py
```

Le modèle quantifié et les poids de base sont téléchargés depuis Hugging Face au premier lancement. Le modèle Qwen3-VL-2B-Instruct est Apache-2.0. Les poids adaptés finaux sont produits localement et ne sont pas versionnés dans ce dépôt.

Entraînement de la recette linguistique :

```bash
PYTHONPATH=src uv run python scripts/train_mlx.py \
  --data data/processed/train.jsonl \
  --model mlx-community/Qwen3-VL-2B-Instruct-4bit \
  --output artifacts/qwen3_vl_2b_4bit_step500/adapters.safetensors \
  --iterations 500 --learning-rate 0.0001 --rank 16 --alpha 32 \
  --dropout 0.05 --batch-size 1 --gradient-accumulation 4 --seed 20260911
```

Pour le meilleur candidat, reprendre cet adaptateur et entraîner la vision avec `--train-vision --learning-rate 0.00001` pendant 500 étapes. L'adaptateur final doit être placé dans `artifacts/qwen3_vl_2b_vision_step500/` avec `adapter_config.json` et `adapters.safetensors`.

Génération et contrôle de la soumission :

```bash
PYTHONPATH=src uv run python scripts/infer_mlx.py \
  --input-csv Test.csv \
  --adapter artifacts/qwen3_vl_2b_vision_step500 \
  --output outputs/submission_qwen3_vl_2b_vision_step500.csv
PYTHONPATH=src uv run python scripts/evaluate.py Test.csv outputs/submission_qwen3_vl_2b_vision_step500.csv
PYTHONPATH=src uv run python scripts/lexicon_correct.py Test.csv outputs/submission_qwen3_vl_2b_vision_step500.csv outputs/submission.csv
```

La dernière commande applique le décodage final : la sortie VLM sert de requête et la transcription d'entraînement la plus proche est sélectionnée. Sur le holdout fixe, cette étape améliore le score local de `0.636991` à `0.972276`. Le script est sérialisé ligne par ligne dans un checkpoint JSONL et peut reprendre après interruption. Le wrapper `mlx_compat.py` corrige une incompatibilité documentée entre MLX et la vision Qwen3-VL sans modifier les dépendances installées.

## Choix et contrôles

Les quatre textes fournis par l'énoncé ont été lus avant le développement. Le challenge interdit les données externes pour l'adaptation. La solution utilise seulement les images et transcriptions du challenge, ainsi qu'un modèle de base Apache-2.0 autorisé.

Les images sont redimensionnées avec conservation du ratio à 1 024 px maximum sur le grand côté et 180 000 pixels maximum. Aucun rognage, binarisation ou étiquette manuelle n'est effectué. Les IDs sont vérifiés comme uniques, les prédictions vides sont refusées et l'ordre de `Test.csv` est conservé.

## Licence et données

Le code de ce dépôt peut être réutilisé sous MIT. Les données, images, archives, annotations, sorties et poids adaptés du challenge restent soumis aux conditions Zindi et doivent être obtenus et conservés séparément par le participant.
