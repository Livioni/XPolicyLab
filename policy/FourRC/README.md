# FourRC: RoboTwin Stage2 evaluation

Eval-only adapter for the local 4RC `stage2_sliding_window_inference.py` API,
using the index-based Stage2 interface introduced by upstream commit `f053eed`.
Supported embodiment: **aloha-agilex**, XPolicyLab profile **arx_x5**, action type **ee**.
One environment per server; `eval_batch` is unsupported. Training and dataset conversion remain in 4RC.

## Setup and quick start

Use the existing `4rc` environment with upstream 4RC installed, and a separate `RoboTwin` simulator environment.
The default upstream source is the sibling `4RC/` directory. Set `FOURRC_ROOT` to override it.
A checkpoint directory must contain the complete Stage2 `config.json` and `model.safetensors`.
The upstream inference loader automatically selects the global visual input from the saved projection shape:
`cross_frame` uses the original 1536-channel features; `concat` uses 3072 channels of concatenated
frame-local and cross-frame features. Both preserve the visual token count and load all weights strictly.
No checkpoint conversion or evaluation flag is needed. TCP-only checkpoints without a global encoder
remain unsupported. New upstream training still defaults to `concat`.
The launcher default checkpoint is set by `checkpoint=...` in `scripts/eval_4rc.sh`; `--checkpoint` overrides it.

```bash
conda activate 4rc
bash XPolicyLab/policy/FourRC/install.sh
# Run from the RoboTwin root:
bash scripts/eval_4rc.sh --dry-run
bash scripts/eval_4rc.sh --task click_bell --test-num 2
# Replan after 4 executed actions (history stays 8, prediction stays 16):
bash scripts/eval_4rc.sh --task click_bell --test-num 2 --execute-horizon 4
bash scripts/eval_4rc.sh --task place_dual_shoes --setting random --test-num 1
# All tasks, same single-environment policy interface; one server per GPU:
bash scripts/eval_4rc.sh --config env_cfg/eval/all_tasks.yml --gpu-ids 0,1 --test-num 100
```

Set `execute_horizon: 8` in `policy/FourRC/deploy.yml`, or pass `--execute-horizon N` to
`scripts/eval_4rc.sh`. Valid values are integers **1–16**, default **8**. The command-line option
sets `FOURRC_EXECUTE_HORIZON` for the launched policy server and overrides the YAML value;
the environment variable can also be set directly. Each inference still predicts 16 poses and
uses 8 RGB frames. `sampling_steps` is the model's flow integration count, not the execution length.
Restart an existing server after changing its configuration.

Use `--checkpoint /absolute/path/to/checkpoint` for another 8-history/16-future Stage2 model.
`--policy-conda-env` and `--eval-env-conda-env` override environment names.
The deployment config defaults `t5_model` to `google-t5/t5-base`, overriding training-machine
paths such as `datasets/t5-base-local` saved in checkpoints. Set `FOURRC_T5_MODEL` to an
absolute local T5 directory or another model ID if needed; it takes precedence over the YAML.
An already cached T5 also works with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.
The policy server must have access to the RoboTwin aloha-agilex assets and `convert_robotwin_tcp.py`
for the training-label TCP definition and measured-joint forward kinematics.

## Observation, time, and query contract

- Uses `vision.cam_third_view.color` RGB, language, calibration, and measured arm joint positions.
  The camera must originate from **observer_camera**; the legacy front-camera alias is rejected.
- Input RGB must be `uint8 [240,320,3]`. No channel swap or resizing. Reflect-pad `(1,1,6,6)`
  to `[252,322]` and normalize to `[-1,1]`. Shift intrinsics and query pixels by `(1,6)` once.
- Required metadata: `additional_info.third_camera_source=observer_camera`,
  `T_world_from_robot_base [4,4]`, and `waypoint_index` (0 initially, then one per executed waypoint).
  Calibration uses runtime `intrinsic_matrix` and `extrinsics_matrix` (world→OpenCV camera).
- **Every window's initial query uses FK of that window's FIRST frame's measured joints.**
  `state.left_arm_real_joint_state` and `state.right_arm_real_joint_state` contain six angles in radians,
  read from `get_*_arm_real_jointState()`. The existing `*_arm_joint_state` fields contain drive targets
  and are not used for query initialization. Missing measured joints fail explicitly.
- FK uses the fingertip contact midpoint that generated `TCP_third`. Project through the actual robot root
  and that same historical frame's camera calibration. Order is always `[left, right]`, regardless of image order.
  No default-home initialization, recovery propagation, or recovery-only bridge remains in the adapter.
- Only the initial two query points of each window use FK. The upstream 4RC reconstructs the window and
  samples its eight-frame historical features from recovered TCP projections, as its inference API specifies.
- Bootstrap repeats the initial RGB eight times, matching the updated Stage2 training sampler.
  The model internally encodes history positions `[-7,...,0]` and future steps `[1,...,16]`;
  `infer_stage2_window` receives no `frame_times` or `frequency_hz`.
- Predict 16 steps, execute steps 1–N (`execute_horizon`), and receive one observation **after each waypoint**,
  including the final observation on success or the action limit. The latest eight observations form the next input;
  startup repeats remain until enough real observations arrive. Window-boundary observations are never duplicated.
- Out-of-image FK queries are clipped to the image bounds as in training's initial-query fallback;
  behind-camera/non-finite queries fail the episode.
- The evaluator executes the selected prefix **one waypoint at a time** via `take_action(..., action_type="ee")`.
  Each arm plans from its current measured joint state, then executes before the next waypoint is planned.
  The original per-arm failure handling applies: a failed arm plan skips that arm's movement for the step;
  gripper control and the other arm continue, and the evaluator collects the resulting observation.
- **Stage2 predicts ordered actions without an execution-frequency assumption.** Each waypoint's
  duration varies. The launcher retains `--frequency 15` as shared evaluator metadata only;
  the adapter neither requires nor passes it to the model. Per-waypoint observations, gripper
  interpolation, success checks and early stopping remain enabled.
- Collection's `4rc_clean.yml: save_freq: 15` is a **physics-step stride**, not an Hz setting.
  At the default 250 Hz simulation timestep, interior samples are 60 ms apart (~16.67 Hz),
  with extra captures at motion boundaries. The current exporter nevertheless writes nominal frequency 15.
  Evaluation uses the permitted **one observation per completed TCP waypoint** fallback because waypoint
  durations vary: the latest eight frames form the input, with initial-frame repetition only during startup.
  No synthetic timestamps are constructed for model conditioning.

## Coordinate and action contract

With `T_A_from_B` mapping B coordinates to A, each complete prediction is converted using the last historical camera:

```text
T_base_from_tcp = inverse(T_world_from_base) @ inverse(T_camera_from_world) @ T_camera_from_tcp
```

The base frame is the common aloha-agilex URDF root, not a camera frame or an individual arm's internal planning frame.
Both translation and orientation are transformed. The logged TCP trajectory is in base coordinates, in metres.

The training-label TCP is the midpoint of the fingertip contact faces, not RoboTwin's built-in TCP getter.
For the validated assets its orientation equals the ee control orientation and its local offset is
`[0.147335, -0.0000015, -0.000713515]` metres (computed from URDF rather than copied into the adapter):

```text
p_ee_base = p_tcp_base - R_tcp_base @ offset
T_world_from_ee = T_world_from_base @ T_base_from_ee
```

RoboTwin's existing `take_action(..., action_type="ee")` expects **world** coordinates and performs base conversion/IK
internally. Thus the adapter converts the base trajectory to world ee control poses at this boundary, without changing
RoboTwin's control API. Each step returns `left_ee_pose`, `right_ee_pose` as `[x,y,z,qw,qx,qy,qz]`, plus scalar-array
`left_ee_joint_state`, `right_ee_joint_state`. Grippers are binary: 1=open, 0=closed; the upstream future score threshold is 0.

Invalid predictions terminate the episode and count as failures. Simulator errors before the first action retain the
existing retry behavior. `reset()` clears the in-memory history; `prepare_case()` supplies the episode seed and metadata.

## Outputs and remote serving

Evaluation results and rollout videos share the same run directory under **`eval_result/rollout/`**:

```text
eval_result/rollout/<task>/<policy>/<task_config>/<checkpoint>/<timestamp>/
  _result.txt              # Evaluation statistics and video filename pattern
  episode0.mp4             # Head camera
  episode0_third_view.mp4  # Third camera
```

Both single-environment and batch evaluations use this directory for `_result.txt` and videos.
No per-waypoint image or NPZ files are written.
Both FourRC task configs enable `eval_video_third_view: true`. The synchronized observer-camera
video (320×240) uses the same RGB source
as the model's third-view input before padding/normalization. Both use the existing 10 FPS
video playback rate; this does not change model or simulator timing. The adapter saves all 16 base-frame TCP poses and grippers per
prediction, execution horizon, historical frame indices, `future_step_indices` (1–16), query points,
calibration, window/seed, and trial outcomes as JSONL under `eval_result/FourRC_predictions/`.
The load event records `time_encoding: index` instead of a model frequency, plus the detected
`global_feature_mode` and `global_visual_input_dim`; the latter two are also printed at startup.
Override that directory with `FOURRC_LOG_DIR`. The first N predicted poses are the ones submitted for execution.

### Waypoint-aligned in-memory history

The adapter retains RGB, measured joints, FK TCP and calibration in memory after each executed waypoint,
including the terminal observation. It does not write per-frame JPG/PNG/NPZ files or history archives.
For example, after executing four waypoints, the next eight-image input is `[0,0,0,0,1,2,3,4]`;
after four more it is `[1,2,3,4,5,6,7,8]`, whose initial query uses **frame 1's** measured joints.
Frame indices count waypoint calls, not physics steps or fixed 15 Hz captures.

### Live point cloud and TCP visualization

Enable the manual inference UI during evaluation:

```bash
bash scripts/eval_4rc.sh --task adjust_bottle --test-num 1 --setting clean --execute-horizon 8 \
  --viser --viser-port 8020 -- --stream-output
```

Open **http://127.0.0.1:8020** on the policy-server host. The actual URL is printed as
`[FourRC viser] ...`; viser chooses another available port if the requested one is occupied.
The viewer closes when the local evaluation stops its policy server. Nothing is written per frame.
It needs `viser>=1.1,<2` in the policy conda environment (included in `install.sh`).

The viewer starts paused before the first model inference. Wait for the blue **推理并执行一次** button
to become enabled, then click once: predict 16 TCP steps, update the scene, execute the first
`execute_horizon` actions, and pause again. The button stays disabled during inference/execution;
extra clicks do not queue more actions. Initial RGB is visible before the first click; reconstructed
point clouds appear after inference. Waiting and browsing do not advance the simulator.
The evaluator polls readiness with short RPCs, so a long manual pause does not hit the 120-second
policy request timeout. If the manual UI fails, evaluation stops instead of falling back to automatic actions.

The viewer contains:

- A **left-docked third-view RGB panel**, synchronized with the selected historical frame.
  TCP overlays use that frame's original camera intrinsics/extrinsics: `L/R FK` mark measured TCPs
  (left yellow / right green), and `L/R +N` mark the selected future step (left orange / right blue).
  Optional `REC` markers show the recovered historical TCPs. Geometry toggles control the overlays;
  points behind the camera or outside the image are omitted. Overlays never modify model input RGBs.
- All eight historical third-view RGBs and their **4RC-reconstructed metric-depth point clouds**, using the same
  `make_point_cloud` function as upstream Stage2 inference. RGB and depth padding are aligned before
  unprojection; invalid points and the lowest 2.5% confidence are filtered, with depth capped at 3 m.
  The default point-count downsample is **2**: project at the native 320×240 resolution, then keep
  every second valid point after filtering (**about half the points**, at most 38,400 per frame).
  Camera intrinsics stay at native resolution. Default point size is **0.003 m**.
- All 16 predicted fingertip TCPs: left orange, right blue. Thick lines highlight the first
  `execute_horizon` commands. Select `Future step` to inspect a predicted orientation and gripper state.
- Measured FK history and post-action TCPs: left yellow, right green. These update after each executed
  waypoint, including the terminal observation, so actual execution can be compared to that window's prediction.
- Optional recovered historical TCPs, plus the robot-base and third-camera coordinate axes.
- `Window` selects an earlier inference. `历史帧 (0–7)` and `上一帧` / `下一帧` select its historical RGB
  and point cloud; the information panel shows the actual waypoint index, sampling fraction and
  displayed point count. Each window keeps its own
  eight reconstructions, including overlapping/repeated observations. Browsing does not trigger inference.
  Future TCPs remain anchored to that window's last frame while inspecting an older cloud.
- `Follow latest` follows new predictions. Clicking the inference button re-enables it.
  `Third-camera view` restores the selected historical camera viewpoint; dragging the scene lets you inspect other angles.

Everything is drawn in **RoboTwin world coordinates, in metres**. The point cloud and recovered TCPs
use the true per-frame camera extrinsics; future TCPs reuse the exact `base_tcp` trajectory used by
`convert_actions`, transformed to world using the true robot-root pose. The display shows the fingertip
TCP rather than the offset control EE. It never feeds geometry back into the policy or changes actions.

Defaults in `deploy.yml`: `viser: false`, `viser_host: 127.0.0.1`, `viser_port: 8020`,
`viser_max_points: 100000` (per historical cloud, applied after point-count sampling), `viser_max_windows: 128`, `viser_manual: true`,
`viser_downsample: 2` (retain about 1/2 of valid points; 1 keeps all), `viser_point_size: 0.003`. Old windows are evicted at the limit, and the
history resets between episodes. All history remains in memory, with no per-frame files.
Configuration can be overridden with `FOURRC_VISER=1`, `FOURRC_VISER_HOST`, `FOURRC_VISER_PORT`,
and `FOURRC_VISER_MANUAL`. `--viser-host` / `--viser-port` also enable the viewer.
Use **`--viser-auto`** (or `FOURRC_VISER_MANUAL=0`) for the previous automatic inference behavior.

For a remote policy host, use `--viser-host 0.0.0.0` and open `http://POLICY_HOST:8020`, or keep the
loopback binding and use `ssh -L 8020:127.0.0.1:8020 POLICY_HOST`. If the server is already running,
set these options when starting that server; client-side flags cannot reconfigure it.
With multiple policy servers, each viewer has its own URL in that task's log.

### Remote serving

Start a server on the policy host (from its RoboTwin root):

```bash
FOURRC_EXECUTE_HORIZON=4 bash XPolicyLab/policy/FourRC/setup_eval_policy_server.sh \
  RoboTwin click_bell /absolute/checkpoint/path arx_x5 ee 0 0 4rc 18080 0.0.0.0
```

On the simulator host:

```bash
bash scripts/eval_4rc.sh --task click_bell --test-num 2 -- \
  --enable-remote --policy-server-ip SERVER_IP --policy-server-port 18080
```

For an already running remote server, set `execute_horizon` in its server-side YAML or set
`FOURRC_EXECUTE_HORIZON` when starting it, as above. A flag on the simulator host cannot reconfigure
a server on another machine.

A remote server still needs this adapter, the upstream 4RC source, checkpoint, and RoboTwin robot assets.
Do not connect simultaneous environments to one stateful server. Multi-GPU scheduling creates separate servers.

See [the earlier training/evaluation audit](TRAIN_EVAL_AUDIT.md) for arm-order and control measurements.
That audit used the former propagated-query implementation; new evaluations use measured-joint FK initialization.

## Checks

```bash
conda run -n 4rc python -m pytest XPolicyLab/policy/FourRC/tests -q
bash scripts/eval_4rc.sh --dry-run
```

Tests cover the strict index-based inference signature, frequency-independent observations, measured-joint FK queries, camera/base/ee transforms, quaternion ordering, RGB padding, all execution lengths 1–16, rolling eight-frame windows and per-waypoint observations (including terminal frames), viewer coordinate alignment, point-count downsampling, click gating, historical-frame browsing and memory bounds, reset, invalid predictions, and the distinction between initial policy failures and simulator failures.
The generic demo debug client does not supply the required third-camera/root calibration; use calibrated observations
for transport testing rather than relaxing validation.

