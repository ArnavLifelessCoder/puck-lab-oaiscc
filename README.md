# Puck Lab

Source repository: https://github.com/ArnavLifelessCoder/puck-lab-oaiscc

## Start here

- **Run the demo (host site):** `start.ps1` or `python serve.py`. Opens host view at http://127.0.0.1:8766/ (improved model, physics reference, Results).
- **Run the student site locally:** `start-student.ps1` or `python serve.py --student`. Rebuilds and serves `dist-student/` at http://127.0.0.1:8767/ with no host models or pages. It also strips the improved-model and physics-reference code from the scripts. Deploy the contents of `dist-student/` for students.
- **Send to teams:** `packages/student-kit.zip`. One laptop per team; no ChatGPT Work subscription required.
- **Read without a Markdown viewer:** PDFs of the guides are next to the `.md` files (`host/PROJECTED_WORKSHOP_RUNBOOK.pdf`, `student/TEAM_BRIEF.pdf`, `student/EXPERIMENTS.pdf`). Rebuild them after editing with `python scripts/make_pdfs.py` (needs `pip install markdown` and Edge or Chrome).
- **Run the workshop:** `host/PROJECTED_WORKSHOP_RUNBOOK.md` (projected room, only the hosts have ChatGPT Work, no attendee laptops). Slide copy is in `host/ACTIVITY_DECK_COPY.md`.
- **Edit the interface:** `app/`. Default is the **student view**; add `?host=1` for **host view** (improved model, physics reference, Results tab).
- **Site for students:** `python scripts/build_student_site.py` writes `dist-student/`, which leaves out the host weights and host-only pages.
- **Debug and submit a repair:** `student/`.

## Folder map

| Folder | Contents |
|---|---|
| `app/` | Browser page, styles, inference engine and planning worker |
| `student/` | Notebook, editable repair, Python lab, team brief and requirements |
| `student/results/` | Generated participant reports; not distributed in the ZIPs |
| `host/` | Reference controllers, scoring script, protocol, runbook and confirmation scenes |
| `host/results/` | Development/confirmation measurements and diagnosis |
| `training/` | Neural training, reference calibration and training dependencies |
| `assets/` | Shared model weights, training observations, fixtures and public summaries |
| `scripts/` | Benchmarking, diagnosis and ZIP-building utilities |
| `tests/` | JavaScript verification |
| `docs/` | Verification record |
| `packages/` | Ready-to-share student and host ZIPs |
| `deployment/` | Isolated Sites checkout containing only the published demo and student download |

Shared assets stay in one place. Keep the folder structure intact when copying or unzipping.

## Commands

From this root folder:

```powershell
python serve.py
node tests/engine.test.mjs
node tests/playback.test.mjs
python training/train.py
python training/calibrate.py
python scripts/diagnose.py
python scripts/benchmark.py development
python scripts/package.py
```

For participants, open a terminal in `student/`:

```powershell
python -m pip install -r requirements.txt
python -m jupyterlab team-notebook.ipynb
python check_submission.py
```

Host final scoring, from the root:

```powershell
python host/score_submission.py path/to/team/student/repair.py
```

Read submitted code before executing it. Final confirmation is for frozen submissions, not iterative tuning. See the host runbook for the full rules.

## Current neural comparison and participant workflow

On a new frozen 96-scene comparison: original NN 1/96, previous improved NN 73/96, new contact-local NN 96/96. All use the same planner. The new model reuses the previous model's training observations, with learned responses in surface-relative coordinates. Known geometry is a supplied prior. This result does not guarantee success outside the fixed deterministic simulator.

Teams start from the weak model. In student/, run:

```powershell
python -m pip install -r training-requirements.txt
python run_experiment.py --name baseline
python run_experiment.py --name contacts --contact-weight 5 --compare baseline
```

Each saved run contains network.json, training metadata and scores. Load its network.json using the website's team-model control. See student/EXPERIMENTS.md for the actual edits and hypothesis cards. Host reference weights remain outside the student ZIP.

Verify browser behavior from the root:

```powershell
node --test tests/engine.test.mjs tests/playback.test.mjs tests/improved.test.mjs tests/local.test.mjs
```

The current host training entry point is training/train_local.py. Evidence and limitations are in docs/NEURAL_REPAIR.md and host/PROJECTED_WORKSHOP_RUNBOOK.md.
