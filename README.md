# Privacy engineering: interactive research demos

<p>
  <img src="src/catdemos/common/static/ieeta.svg" alt="IEETA" height="56">
  &nbsp;&nbsp;
  <img src="src/catdemos/common/static/ua.svg" alt="Universidade de Aveiro" height="56">
</p>

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Stack](https://img.shields.io/badge/FastAPI-REST%20%2B%20WebSocket-orange)

Four live demos of the privacy-engineering research of **Catarina Silva** (IEETA, University of Aveiro).
Each demo asks one question from a paper and answers it with the paper's method running on a server.
Every mouse movement is sent over a WebSocket, the server reruns the model on real data, and the page
redraws. The demos are simplified versions of the papers, but none of them is a mock-up.

| Demo | Question | You do | The server runs |
|---|---|---|---|
| **ASAP Latent Map** (`:8001`) | Can a privacy-harmful Android app be spotted from the permissions it asks for? | Move over a map of 29,332 real apps; drag the detection threshold | Autoencoders decode and judge the permission profile under the mouse; confusion matrix on 7,333 held-out apps |
| **PsDC Categorizer** (`:8002`) | How does PsDC categorize the data an app collects, and what is that data worth? | Drag fields into an app's data model; set how the data is collected, kept and how precise it is | Automatic categorization of each field; profiler retrained to measure what the combination reveals; privacy value |
| **DP vs Profiling** (`:8003`) | Differential privacy stops re-identification, but does it stop profiling? | Choose which fields to hide with differential privacy; move along the privacy budget ε | Local-DP release of the chosen fields; profiler retrained and linkage attack rerun at every ε |
| **Elastic Privacy Map** (`:8004`) | Can a phone hide a patient's whereabouts every day, yet stay precise during a crisis? | Draw a walk around Fábrica Centro Ciência Viva and the UA campus (or load a routine walk or one with a crisis); pause or scribble to act out a crisis | 1 Hz trace, autoencoder crisis detector, neuro-evolved governance agent, planar-Laplace membrane |

A start page on `:8000` links the four and shows whether each one is running.

## Quick start

```bash
./run.sh              # all five servers; or: ./run.sh asap dp
```

`run.sh` finds Python ≥ 3.11 and creates `./venv`, then installs the dependencies. On the first run it
also downloads the datasets (about 6 MB) and trains the models (about 1 min). Later runs load everything
from `./data` in under a second. It then starts the servers and waits for each to answer `/health`,
prints the URLs, and stops everything on Ctrl-C. Set `HOST=0.0.0.0` to serve other devices on the LAN.
Each server gets 2 native math threads by default (`CATDEMOS_THREADS`), overriding `OMP_NUM_THREADS`
and the like for the servers only. With five servers, one thread per core would oversubscribe the CPU
and make each measurement 2–3× slower.
The script runs on the bash 3.2 that ships with macOS.

Needs an internet connection for the first run and for the map tiles (OpenStreetMap) in the Elastic demo.

## The demos and how close they are to the papers

### ASAP Latent Map
*ASAP 2.0: Autonomous & Proactive Detection of Malicious Applications for Privacy Quantification in 6G
Network Services* (Computer Communications, 2025); *ASAP* (FiCloud 2024); *Assessing Mobile Application
Privacy* (2023).

- **As in the paper:**
  - unsupervised autoencoders on Android permission vectors;
  - the anomaly score is the reconstruction error;
  - a dynamic threshold (the MCC-optimal one on a validation split);
  - evaluation by MCC on held-out apps.
- **Simplified:**
  - One dataset, NATICUSdroid (UCI #722, 29,332 apps × 86 permissions), instead of three.
  - A benign-only autoencoder reaches MCC ≈ 0.54 here, so the score compares a benign and a malicious
    autoencoder (log error ratio). That gives **MCC 0.886** (TPR 0.93, FPR 0.04). A supervised classifier
    reaches ≈ 0.93 on the same split; the paper reports 0.976 on its own datasets.
  - The navigable map comes from a third autoencoder with a 2-D bottleneck.
  - The A–E label follows the idea of a qualitative privacy label (SCALE, 2023/2025). Its weighting
    (60 % detector probability, 40 % exposure of the permissions, weighted by PsDC tier) is this demo's own.

### PsDC Categorizer
*Proactive Data Categorization for Privacy in DevPrivOps* (Information, 2025); *Semantic and Numerical
Feature Clustering for Automated Privacy Quantification* (FiCloud 2025).

- **As in the papers:** the data model is described on PsDC's three levels:
  - *direct data categorization*: each field gets a category;
  - *dynamic tags*: what the combination of fields lets an adversary infer;
  - *structural attributes*: frequency, retention, granularity.
- **Categorization is automatic** and uses both kinds of features from paper 22. Semantic features
  (character n-gram TF-IDF of the field's name and its most frequent real values) are compared with a
  one-line definition of each category (identity, socio-economic, sensitive). Numerical features (mutual
  information of the values with sensitive facts) weigh each field in the score. It agrees with the
  expert categories on all 13 census fields. The expert labels are used only to report that agreement.
- **Dynamic tags are measured, not declared.** For every sensitive fact that was not collected (income,
  marital status, sex), a profiler is retrained on the collected fields; the tag shows its accuracy
  against guessing. The share of the 9,000 people made unique by the collected fields is also counted.
  Coarse granularity bins the values, which really changes the tags.
- **Simplified:**
  - The census (UCI Adult) stands in for an app's data.
  - The privacy value (structural × (½ direct + ½ dynamic), with fixed category weights) is this demo's
    own aggregation.

### DP vs Profiling
*Evaluating the Effectiveness of Differential Privacy Against Profiling* (ICCT-Europe 2025).

- **As in the paper:** DP is applied to released data. Its effect is measured on re-identification and on
  profiling, a model that infers a sensitive fact.
- **Simplified:**
  - One dataset (UCI Adult: 6,000 records to train, 3,000 to test).
  - Local DP per chosen field: Laplace noise for numbers, randomised response for categories, ε per field.
  - Logistic regression as the profiler.
  - Re-identification is a nearest-neighbour linkage of 800 people.
- **Result, reproduced live:**
  - With every field hidden at ε ≈ 1.8, profiling keeps 50 % of its advantage over guessing, while
    re-identification keeps 2 %.
  - Release `relationship` exactly and hide the rest: only 3 % of people can be re-identified, yet
    "married" is still guessed at 98 %.
  - This is the paper's point: DP stops linkage, not profiling.

### Elastic Privacy Map
*Self-Adaptive Governance for Elastic Privacy in Mental Health Digital Phenotyping* (FiCloud 2026).

- **As in the paper:**
  - an autoencoder crisis detector (anomaly detection on behaviour);
  - a neural agent optimised by neuro-evolution (here the cross-entropy method) that sets the privacy
    membrane;
  - synthetic crises injected into routine behaviour;
  - evaluation as privacy shield (routine) against spatial error (crisis).
- **Simplified:**
  - Routine walks are simulated on the real Aveiro street network (OpenStreetMap, 5,750 nodes), with
    phone-like GPS error (AR(1), σ = 2 m).
  - Crises are pacing, freezing and wandering.
  - The detector's autoencoder is a dense MLP, without a sparsity penalty.
  - Obfuscation is planar-Laplace geo-indistinguishability, with mean offset equal to the membrane radius
    (5–200 m).
- **Held-out benchmark (80 walks):**

  | | this demo | paper |
  |---|---|---|
  | privacy shield, routine | 180.4 m | 117.4 m |
  | spatial error, crisis | 17.8 m | 9.9 m |

  The demo also detects 93 % of crisis seconds, with 10 % false alerts in routine. Most false alerts are
  ordinary waits (at a crossing, for instance), which the detector cannot fully tell from freezing. The
  agent therefore also sees a 60 s average of the anomaly score, which cut false alerts from 14 % to 10 %.
  "Routine walk" loads a walk with no crisis, so any alert in it is a false alarm, and the page says so.

## Architecture

```
run.sh ─┬─ hub      :8000  start page, polls each demo's /health
        ├─ asap     :8001  ┐
        ├─ psdc     :8002  │
        ├─ dp       :8003  ├─ one FastAPI process per demo
        └─ elastic  :8004  ┘
```

Every demo server follows the same pattern (`src/catdemos/common/server.py`):

- **REST** for setup and structured data: `GET /api/meta` or `/api/overview`, `/api/sample`, and a REST
  mirror of every live operation (handy for scripting and tests). OpenAPI docs are at `/docs`.
- **WebSocket `/ws`** for live updates. The client sends `{"op", "seq", "args"}` and the server answers
  `{"op", "seq", "ms", "data" | "error"}`. REST and WebSocket call the same handler through `Live`, which
  runs it off the event loop, under a lock, because the engines are not thread-safe. Bad arguments
  (missing, not finite, out of range, unknown names) come back as `error`; an unexpected engine error is
  logged and answered with `"internal error"`, and the session stays open.
- **Latest-wins client** (`common/static/client.js`). At most one request per operation is in flight.
  While it runs, newer pointer positions replace older ones, so fast mouse movement never queues work on
  the server. The footer shows server time and round-trip time.
- **Train once, cache.** Models are trained on first start and stored with joblib in `data/cache`. The
  file name carries a version, which is bumped whenever the training code changes. Seeds are fixed.
  If the OSM street graph cannot be fetched, Elastic trains on a regular grid for that run only and does
  not cache it, so the next start retries OSM.
- Viewers are plain ES modules with no build step. The only third-party script is Leaflet, from cdnjs.
  Colours: the IEETA palette (`#FF8145`, `#142441`, `#293490`, `#F3F6FD`), filled in with Nord.
- Branding (`common/static/`): `ieeta.svg` and `ua.svg` in colour, `*-white.svg` reversed for the navy top
  bar (IEETA left, links to the start page; UA right), and `favicon.svg`, a simplified IEETA hand mark
  with thicker strokes so it stays legible at 16 px.

```
src/catdemos/
  common/   server.py (app factory, Live, CLI), datasets.py (download-once loaders),
            static/ (theme, client, logos, favicon)
  asap/     model.py (autoencoders, map, labels), app.py, static/
  common/   adult.py (census sample, DP mechanisms, profiler, linkage, uniqueness), shared by psdc and dp
  psdc/     categorize.py (automatic categorization, dynamic tags, privacy value), app.py, static/
  dp/       study.py (DP on chosen fields: profiling vs re-identification), app.py, static/
  elastic/  osm.py (Aveiro street graph), pipeline.py (simulation, features, detector, agent, membrane), app.py, static/
  hub/      app.py, static/
  demos.py  registry: ports, questions, papers
research/   paper review, candidate demos and their scoring (how these demos were chosen)
```

## Performance

Server compute per interaction, measured after warm-up on a laptop CPU (AMD Ryzen AI 7, single request);
an M2 MacBook Air is in the same class:

| Operation | Time |
|---|---|
| ASAP probe (decode + two detectors + 7-NN) | 0.6 ms |
| ASAP threshold (confusion matrix on 7,333 apps) | 0.05 ms |
| PsDC data model (categorize + 2–3 profiler fits + uniqueness), first time | 20–30 ms, then cached |
| DP point (release + profiler fit + linkage on 12 fields), first time | 30–100 ms, then cached |
| Elastic walk, 12 min / 1 h of walking | 3.5 ms / 14.5 ms |

## Development

```bash
python -m venv venv && venv/bin/pip install . --group test
CATDEMOS_DATA=data PYTHONPATH=src venv/bin/python -m unittest discover -s test
pre-commit install        # every commit runs the CI checks locally
```

`.pre-commit-config.yaml` and `.github/workflows/ci.yml` run the same checks: pre-commit-hooks hygiene,
`ruff check`, `ruff format --check`, `node --check` on the viewer scripts, `basedpyright`, `vulture`,
`unittest`, and coverage with a ratcheting floor. ruff, basedpyright and vulture are system tools, pinned
in CI to the local versions; they are not project dependencies.

Useful single commands: `python -m catdemos warmup` (data and models only),
`python -m catdemos.asap --port 9001` (one server).

## Data and licences

| Source | Used by | Licence |
|---|---|---|
| NATICUSdroid, UCI ML Repository #722 | ASAP | CC BY 4.0 |
| Adult, UCI ML Repository #2 | PsDC, DP vs Profiling | CC BY 4.0 |
| OpenStreetMap (Overpass API, tiles) | Elastic | ODbL, © OpenStreetMap contributors |

Datasets are downloaded to `data/raw` on first use and are not part of the repository.

The IEETA and Universidade de Aveiro logos belong to their institutions and are not covered by the MIT
licence. The SVGs were traced with potrace from the PNG logos published on ieeta.pt.

## Citation

See [`CITATION.cff`](CITATION.cff) for the software and the papers it implements.

## Licence

MIT, see [`LICENSE`](LICENSE).
