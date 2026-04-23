# DRISHTI Django Console

This Django app is the product surface for the DRISHTI prototype.

## What it does

- accepts a suspicious video upload
- runs the current deepfake pipeline
- enriches the result with watchlist and weaponisation layers
- produces an analyst-facing response dashboard
- generates a downloadable text report

## Run locally

```bash
cd "Django Application"
pip install -r requirements.txt
python manage.py runserver
```

Open:

```text
http://127.0.0.1:8000/
```

## Notes

- If PyTorch checkpoints are unavailable, the app still runs in demo fusion mode.
- The templates are now tailored to the DRISHTI defence-tech narrative.
- Logs are written under `Django Application/logs/` as JSONL when detections or feedback are submitted.
