# FORKCAST — Multiverse of Monsoon

**Fork the Forecast. See the Impact.**

FORKCAST is an experimental, offline counterfactual weather-simulation prototype for the India 32 × 36 daily climate grid. It runs a baseline forecast and an alternate forecast after a localized physical perturbation, then compares the two rollouts.

> This is a research/demo prototype, not an operational weather-warning or agricultural decision system.

## Current implementation

- **src/dataset.py**: indexed NetCDF data access, chronological splits and training climatology.
- **src/model.py**: compact local residual CNN; physical-unit inputs/outputs; 14-step model horizon.
- **src/train.py**: one-step and autoregressive training, checkpoint resume, telemetry and memory guard.
- **src/evaluate.py**: persistence/climatology comparisons, rainfall CSI, physical validity and shock diagnostics.
- **tests/test_forkcast.py**: unit and contract-oriented tests.
- **serve.py**: offline FastAPI inference endpoints (POST /health, POST /simulate).
- **dashboard/**: browser dashboard; every forecast displayed is requested from the live API.
- **stats.json**: official six-channel normalization statistics.

## Official interface

The server follows the Round 1 contract:

- Start command: **python serve.py --port 8000**
- Binds to **0.0.0.0**, one worker, CPU only, at most four PyTorch threads.
- **POST /health** reports ready only after a trained checkpoint is loaded.
- **POST /simulate** accepts physical-unit nested lists:
  - **init_states**: [3, 6, 32, 36], oldest day first.
  - **doy_sincos**: [sin, cos] for the last input day.
  - **steps**: integer from 1 through 14.
  - **perturbation**: null or [6, 32, 36], physical delta added once to the last input state.
- Response: **rollout** [steps, 6, 32, 36] in physical units and **inference_ms**.
- Channel order: **t2m** (K), **tp** (mm/day), **u10** (m/s), **v10** (m/s), **msl** (Pa), **tcwv** (kg/m²).

The official contract and validator are in the organiser kit:  
https://github.com/OfficialAbhinavSingh/yug-monsoon-kit  
https://github.com/OfficialAbhinavSingh/yug-monsoon-kit/blob/main/docs/R1_CONTRACT.md

## Data and checkpoint placement

Do not commit the raw historical NetCDF archive or training run directories.

Expected local training files:

- C:\FORKCAST_DATA\india_climate_train.nc
- C:\FORKCAST_DATA\stats.json
- Training checkpoints under C:\FORKCAST_DATA\runs\...

For inference/submission, place the selected trained checkpoint at **weights/model.pt**.

The checkpoint must contain **model_state_dict**, including the model's **mean**, **std**, **static_raw** and **static_preprocessed** buffers. The server reconstructs the model and static inputs from those embedded buffers; it does not read the raw NetCDF file at startup.

**Current readiness:** weights/model.pt is not included in this source-only commit. Until a validated trained checkpoint is copied to that path, /health and /simulate return HTTP 503 by design. The server does not substitute a persistence forecast.

## Local setup (Windows PowerShell)

Python 3.12 is recommended.

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

Place the data files at the paths above. Install the organiser kit v1.0.0 wheel locally so the official validator and repo scanner are available.

## Training

See the CLI options before running:

~~~powershell
python -m src.train --help
~~~

Example full training command (review options and data paths first):

~~~powershell
python -m src.train --nc C:\FORKCAST_DATA\india_climate_train.nc --stats C:\FORKCAST_DATA\stats.json
~~~

To resume from a checkpoint, use **--resume-ckpt** and give a separate **--out-dir** so earlier runs are preserved. Keep all training artifacts outside the Git repository.

## Tests and validation

~~~powershell
python -m unittest discover -s tests -v
python -m src.train --smoke-test
python serve.py --port 8000
~~~

In a second terminal, run the organiser validator:

~~~powershell
python -m monsoon_r1.schema_validator http://localhost:8000
~~~

Run the organiser repository scan before submission. Also rehearse with the repository mounted read-only and networking disabled; runtime dependencies, weights and static data must already be present.

## Dashboard

Start the server, then open:

- http://localhost:8000/

The dashboard accepts a JSON file containing one real input window (init_states and doy_sincos), calls the live /simulate endpoint, and displays baseline, shocked and difference grids. It does not contain pre-rendered forecasts.

To create a valid starting-window JSON, export a real three-day sample from the supplied dataset in the local training workspace. Do not commit historical weather windows or lookup tables to the repository.

## Evaluation and known limitations

The previous local K=1 and K=4 pilot reports showed improvement over persistence, but K=4 still had negative mean skill versus the best reference and did not pass the official outward-shock energy-growth gate for both nonzero shocks. These are pilot results, not final holdout or hidden-test results.

Do not claim qualification, official validator success, or final accuracy without running the current checkpoint against the official scorer and all eight request cases. Keep the 2021 holdout separate until model selection is complete.

## Ground rules

- No external weather APIs, internet calls or downloads at inference.
- No hardcoded dates, historical lookup tables or shipped weather archives.
- No GPU requirement; max four CPU threads and six GB RAM in judging.
- The repository is read-only during judging; do not write caches or bytecode into it.
- Label accumulated rainfall differences as **scenario rainfall exposure**, not predicted flood damage.
- Keep the model experimental and document failure modes.

## Licence and data attribution

The team kit code is MIT-licensed. The dataset attribution and terms are in the organiser kit's NOTICE file. Include the required credits in the final submission.
