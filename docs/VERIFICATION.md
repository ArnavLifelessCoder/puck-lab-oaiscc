# Verification

- 7 notebook cells executed sequentially in Python with plot outputs saved. This check did not launch a Jupyter UI.
- Six JavaScript tests passed: original Python/JS inference and simulator parity, contact mechanics, deterministic nested candidate pools, endpoint scoring, contact-rich parity for both repairs, and reference planning.
- All 4,000 original validation states matched scalar simulator versus vectorized fitted-reference outputs to 1e-12.
- Student scoring reproduced the unchanged baseline's 0/24 development result. The instructor repair passed the same submission interface with 96/96 confirmation successes.
- Fixed comparison: neural 1/96, contact repair 4/96 (one regression), calibrated reference 96/96, privileged oracle 96/96. Full paired evidence lives in host/results/confirmation-results.json. The same frozen benchmark was regenerated without tuning after a report-output filename collision was found and fixed.
- Browser reference test: predicted and actual endpoint miss 0.6%, inside the target. Controller changes, execution and history were exercised.
- CPU timing is local evidence, not a guarantee for attendees' laptops. Full live workshop timing and participant usability still require an in-person rehearsal.

The reference has privileged structural assumptions: known geometry and contact equation form, with drag/restitution fitted from the original observations. It is a physical model, not an improved neural network. Success in this matched deterministic simulator does not generalize automatically to noisy or misspecified environments.

## Folder organization check

Moved interface files to app/, participant code to student/, reference implementation to host/, training code to training/, ZIPs to packages/, and reports into their matching results folders. Updated imports, notebook paths, root HTTP redirect and packaging. All seven notebook cells ran from student/, all six JavaScript tests passed, and participant evaluation still reproduces 0/24 for the unchanged baseline. No model weights or experiment definitions changed.

## Playback freeze regression

Reproduced the former execution handler with a first animation timestamp of 99 and performance.now() start of 100. It calculated frame -1, threw while drawing an undefined puck position, and left busy=true. Playback now uses the first requestAnimationFrame timestamp as its clock origin, clamps the frame index to 0..35, draws the initial state immediately, and reports drawing failures while unlocking controls. Three new app-handler regression tests cover the earlier initial timestamp, a rendering failure, and a delayed completion frame. The six inference/physics tests still pass. No model weights, calibrated coefficients, action selection or physics equations changed.


## Neural improvement update

See NEURAL_REPAIR.md for the current neural comparison: 2/96 original versus 59/96 improved on a fresh frozen comparison, with 37 remaining failures. The previous physical-reference numbers below/above belong to a different scene set. All 12 current JavaScript tests pass, including neural parity and all 24 development selections; all seven student notebook code cells execute with Python/Agg. Jupyter UI was not tested.


## Contact-local model and team experiments

Current fresh confirmation: original 1/96, previous improved 73/96, local neural 96/96. Same data for the two improved models. All 17 JavaScript tests pass, including model upload/error recovery and Python parity. All seven revised notebook code cells execute through named training, 24-scene scoring and report creation (Python/Agg; Jupyter UI not exercised). A same-data, same-seed free-motion neural-head exercise improves the team starter from 2/24 to 13/24 with no losses. Negative trials are preserved. See NEURAL_REPAIR.md and host/results/student-exercise-rehearsal.json.


## Route comparison and equal shot limits

The default 512-candidate scene uses a 0.110 speed cap and 35 steps for all models. Original NN selects speed 0.031187; local neural/reference select 0.106173. The latter actual route length is 2.380264 table widths. Giving that same push to the original network changes its prediction, not the simulator trajectory. The shared objective scores final miss and boundary/bumper penalties, not shortest path. No model-specific route-length restriction exists.

Added Predict same push, predicted/actual route lengths and chosen-speed/limit display. Manual input now respects the selected speed cap; it previously retained a 0.110 maximum even when the planner cap was 0.040. All 19 tests pass, including identical-action execution across models and manual/remembered-action limits. Model weights, planner ranking and frozen benchmark outcomes are unchanged.
