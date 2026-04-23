# DRISHTI

**Real-Time Military Deepfake & Disinformation Detection Engine**

DRISHTI is a defence-tech hackathon prototype built to demonstrate how India could detect, triage, and respond to synthetic military misinformation in real time. Instead of behaving like a generic academic classifier, this repo now presents an operator-first workflow:

- ingest a suspicious clip
- run multimodal deepfake analysis
- score likely impersonation targets from an India-specific watchlist
- estimate weaponisation potential during sensitive military or diplomatic windows
- generate a PIB-style analyst report

## Why this repo exists

During conflict or diplomatic escalation, narrative control can be lost in minutes. Manual fact-checking is too slow when adversaries can push hundreds of synthetic clips per hour. DRISHTI is designed as the software layer that sits between raw incoming media and a government analyst response.

This prototype is shaped around the problem statement:

- **Multimodal detector**: face-swap risk, lip-sync anomaly scoring, metadata forensics, and audio-visual sync proxy
- **India-specific context layer**: watchlists for figures such as Narendra Modi, S. Jaishankar, Indian Army leadership, and ISPR spokesperson channels
- **Viral spread predictor**: reach, emotional manipulation, and timing fused into a weaponisation score
- **Analyst dashboard**: queue-style UI, threat verdicts, watchlist hits, and report generation

## What is implemented

### Web app

The Django app in [`Django Application`](</Users/pius_rego/Any/hack/deepfake-detection-ai-/Django Application>) now behaves like a DRISHTI operator console:

- mission-focused landing page
- suspicious clip intake flow
- fused analysis result screen
- watchlist match panels
- weaponisation predictor
- downloadable analyst report
- feedback loop and ops metrics page

### Analysis pipeline

The current repo supports two paths:

- **Demo fusion mode** when heavy ML dependencies are missing
- **Checkpoint mode** when PyTorch models are available

In both cases, the user experience stays consistent and produces:

- verdict
- confidence
- signal-by-signal breakdown
- impersonation matches
- recommended operator actions

## Quick start

### 1. Install dependencies

At the repo root:

```bash
pip install -r requirements.txt
```

Or inside the Django app:

```bash
cd "Django Application"
pip install -r requirements.txt
```

### 2. Run the console

```bash
cd "Django Application"
python manage.py runserver
```

Then open:

```text
http://127.0.0.1:8000/
```

### 3. Demo it well

For the best hackathon storytelling:

1. Upload a suspicious clip whose filename includes keywords like `army`, `jaishankar`, `ispr`, or `sindoor`.
2. Show the fused threat score and watchlist match panel.
3. Open the generated analyst report.
4. Explain how this compresses detection-to-response time for a PIB / DRDO / BEL style operator.

## Repo structure

```text
.
├── Django Application/       # DRISHTI operator console
├── Model Creation/           # Training and preprocessing notebooks
├── Documentation/            # Legacy academic/project artifacts
├── github_assets/            # Images and media assets
└── README.md
```

## Important note

This is still a hackathon prototype. The current “India-specific embeddings”, “event calendar”, and “weaponisation predictor” are implemented as a demo-ready operator layer on top of the existing project so the repo can convincingly solve the stated problem within limited time.

## Recommended next upgrades

- plug in actual audio models for lip-sync verification
- replace heuristic OSINT scoring with live source ingestion
- add persistent case management and alert routing
- load real embedding libraries for approved public-figure watchlists
- integrate crisis calendars and media-monitoring feeds

## Where to look next

- [`START_HERE.md`](/Users/pius_rego/Any/hack/deepfake-detection-ai-/START_HERE.md)
- [`EXECUTIVE_SUMMARY.md`](/Users/pius_rego/Any/hack/deepfake-detection-ai-/EXECUTIVE_SUMMARY.md)
- [`Django Application/README.md`](</Users/pius_rego/Any/hack/deepfake-detection-ai-/Django Application/README.md>)
