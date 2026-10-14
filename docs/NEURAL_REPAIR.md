# Current contact-local neural repair

On the newly frozen seed-202609292 scene set, original, previous improved and contact-local networks reach 1/96, 73/96 and 96/96 targets respectively. Same action pools, budget, horizon and scoring. The new model retains all 73 previous successes and adds 23. Detailed paired rows and intervals: host/results/local-neural-confirmation.json. All 24 development scenes are also reached. These scores are one model seed on a deterministic environment, not a guarantee of general reliability.

The data is unchanged from the previous improved model: 177,319 training states and 17,713 independent validation states, with matching SHA-256 hashes. Free and wall outputs are learned affine neural layers. A residual 3-32-32-4 MLP predicts bumper responses in normal/tangent coordinates. Geometry-based routing, local coordinate transforms, affine motion branches and residual structure are explicit priors. Response coefficients and nonlinear weights are learned from observations. No simulator, restitution coefficient, reflection formula or contact-position override runs in inference. Coordinates, architecture, seed and optimization changed jointly, so an isolated causal attribution is not supported.

Training: training/train_local.py. Inference: student/local_dynamics.py and app/engine.mjs. Weights/metadata/parity: assets/local-*.json. Protocol and pre-test freeze: host/local-neural-protocol.json and host/local-neural-freeze.json. Prior artifacts remain in host/experiments/gated-neural-v2/. The new 96-scene set differs from the earlier 59/96 set, so compare the three models within each table rather than subtracting scores across tables.

Verification: 17 JavaScript tests pass, covering Python/PyTorch parity, all 24 development choices, raw-input and geometry-input student models, zero-weight contact behavior, malformed team-file recovery, original/reference behavior and playback. The team experiment runner is rehearsed end to end and records negative results rather than hiding them. Each team's JSON checkpoint can be loaded locally in the browser without uploading it to a server.

## Participant rehearsal

A controlled student implementation change, adding a separately fitted affine free-motion neural head, improved 2/24 to 13/24 development targets with no regression. Training observations, seed, 40 epochs and planning settings were unchanged. Collision weighting and basic geometry/data switches did not help at those settings; all results are retained in host/results/student-exercise-rehearsal.json. The run_experiment helper preserves named artifacts and reports changed settings plus paired gains/losses. Teams load the saved weights in the same browser demo.

## Earlier neural iteration, retained evidence

# Neural repair evidence

The main corrected option is now a genuinely trained neural dynamics model. It is a stronger candidate, not a perfect solution.

## Results

| Fixed evaluation | Original NN | Dense neural candidate | Specialized neural candidate |
|---|---:|---:|---:|
| 24 development scenes | 0 | 1 | 16 |
| 96 fresh confirmation scenes | 2 | Not evaluated | 59 |

The final checkpoint was frozen before confirmation. Mean confirmation miss falls from 0.4114777457 to 0.0963671243 table widths. Both original successes remain successes; 57 additional targets are reached, 37 remain missed. See host/results/neural-confirmation.json for every paired outcome and confidence interval. Extra data, features, architecture and optimization changed together, so this is not a one-factor ablation. Only one final seed/checkpoint and one deterministic simulator family were tested.

## Learned mechanism and declared priors

Inputs include position, velocity, tentative wall clearances, bumper clearance and normal. Known geometry routes each state to a free or contact branch. The free branch is a learned affine neural layer fit by least squares. Contact responses use a residual 11-64-64-4 ReLU network trained from observed state changes. Learned deltas are added to the current state. Reflection, restitution, collision projection and ground-truth motion updates do not run inside improved inference. Geometry is an explicit prior, not an unreported discovery by the network.

Training states are generated independently from final planning scenes, with independent validation seed. The 177,319 training and 17,713 validation states contain no identical state duplicates within or across those partitions. Data covers speeds 0 to 0.115 and includes deliberate near-boundary sampling. The contact branch uses 23,336 contact training samples and 600 epochs after the dense neural warm start. The dense first candidate was inadequate and is preserved rather than hidden.

Inference: student/improved_dynamics.py and app/engine.mjs. Training: training/train_improved.py then training/train_specialized.py. Recorded inputs, versions, loss history and hashes: assets/improved-metadata.json. The data budget and architecture differ from the original; lower conditional or aggregate errors on different validation distributions are not a fair direct accuracy comparison.

## Verification

The browser matches independent PyTorch one-step fixtures within 2e-6, and all 24 development action selections and actual outcomes match Python. Zeroing both branch weights produces zero motion even at a collision: no physical-response fallback supplies the answer. Original-model parity, reference-model behavior, deterministic planning and playback regression checks are retained. The planner uses neural rollouts for selection, then the unchanged simulator executes only the selected action.

The student starter is separate from the solved host model. It supports editable neural layers, contact weighting, geometry features and original or expanded data, with explicit train/validation separation and training-only normalizers. It exports portable JSON weights and logs configuration. Its switches are experiments, not guaranteed fixes. The host solution and final scenes stay outside the student ZIP.

Final checks: all 12 JavaScript model/planning/playback tests passed. All seven notebook code cells ran in order with Python and noninteractive Matplotlib, including 40-epoch training, portable export and 24-scene evaluation (starter candidate 2/24). The expanded-data, geometry-feature and contact-weighting training path also passed a two-epoch smoke run and export parity. The Jupyter interface itself was not exercised because this host Python lacks nbformat. Install the supplied training-requirements.txt on participant machines before the event. Browser verification covered improved neural planning, execution, history recording and unlocked controls; the opening 512-candidate shot missed, which is retained as an honest failure.
