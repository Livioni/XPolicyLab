# FourRC: wrist-conditioned RoboTwin Stage2 evaluation

Eval-only adapter for the current local `4RC/stage2_sliding_window_inference.py` API.
Training and data conversion remain in 4RC. Supported robot: **aloha-agilex**, profile
**arx_x5**, action type **ee**. One environment per policy server; `eval_batch` is unsupported.
Only wrist-conditioned, 3072-channel concat, quantile-v1 Stage2 checkpoints are accepted.

## Installation and model assets

Use the existing `4rc` conda environment with upstream 4RC installed and the separate
`RoboTwin` simulator environment. From the RoboTwin root:

```bash
conda activate 4rc
bash XPolicyLab/policy/FourRC/install.sh
```

The upstream source defaults to the sibling `4RC/`; override it with `FOURRC_ROOT`.
The policy server also needs RoboTwin aloha-agilex URDF assets and `convert_robotwin_tcp.py`
for the training-label TCP definition and measured-joint FK.

The default checkpoint is:

```text
checkpoints/4rc-stage2-action-clean-qity-1-task-wrist-40chunk/checkpoint-10000/
  config.json
  action_norm.json
  model.safetensors
```

All three files are required to start a local server. The configuration must specify
`training_stage: 2`, metric geometry, `use_wrist_condition: true`, a saved
`wrist_encoder_config`, and quantile action normalization version 1. Weights must
contain the concat global projection with 3072 input channels and the complete
wrist branch. The upstream loader restores frozen DINOv2 weights from the checkpoint,
so no separate wrist-model download is needed. Its strict loader checks all weights
and verifies that `action_q01` / `action_q99` exactly match `action_norm.json`.
No old-model fallback or weight conversion is provided.

T5 defaults to `google-t5/t5-base`; `FOURRC_T5_MODEL` overrides it with a model ID
or local Transformers directory. Cached T5 assets can be used with
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

## Evaluation

```bash
# Validate metadata and the schedule without loading weights or starting processes:
bash scripts/eval_4rc.sh --dry-run
# Default checkpoint: 4 history frames, 40 predictions, execute the first 8:
bash scripts/eval_4rc.sh --task adjust_bottle --test-num 2
# More frequent replanning:
bash scripts/eval_4rc.sh --task adjust_bottle --test-num 2 --execute-horizon 4
# Execute a full predicted chunk:
bash scripts/eval_4rc.sh --task adjust_bottle --test-num 1 --execute-horizon 40
# Manual inference and geometry viewer:
bash scripts/eval_4rc.sh --task adjust_bottle --test-num 1 --viser -- --stream-output
# Task scheduling, with one independent server per GPU:
bash scripts/eval_4rc.sh --config scripts/eval_4rc_tasks.yml --gpu-ids 0,1 --test-num 100
```

Use `--checkpoint PATH` for another current wrist-conditioned checkpoint.
Relative checkpoint paths resolve against the RoboTwin root. History length K and
prediction horizon H come from the checkpoint, with no fixed 4/40 requirement.
Execution length N defaults to 8 and must be in **1..H**. `--execute-horizon` sets
`FOURRC_EXECUTE_HORIZON`, overriding `deploy.yml`. `sampling_steps: 8` is the flow
integration count, independent of N. Restart the server after configuration changes.
The default checkpoint was trained only on `adjust_bottle`; evaluating other tasks
uses the same interface but does not imply those tasks were trained.

The launcher supports clean/random settings and overrides for the policy and
simulator conda environments. `ROBOTWIN_RT_DENOISER` defaults to `optix` here.
Dry-run requires configuration and normalization files; a normal local run additionally
requires weights. For a remote run, weights are required on the policy host.

## Observation and preprocessing contract

Each waypoint observation must provide decoded RGB `uint8 [240,320,3]` for:

- `vision.cam_third_view.color`, sourced from RoboTwin `observer_camera`;
- `vision.cam_left_wrist.color`, sourced from `left_camera`;
- `vision.cam_right_wrist.color`, sourced from `right_camera`.

Images stay RGB throughout the transport and adapter. Missing wrist images,
wrong sizes or wrong dtypes fail explicitly; no image substitution is performed.
The clean/random task configurations already enable both wrist cameras at this resolution.
Neither sensor depth nor wrist-camera calibration is an input requirement.

Third-view history is stacked as `[1,K,3,252,322]`, reflect padded `(1,1,6,6)`
and normalized to `[-1,1]`. Third-camera intrinsics and query pixels are shifted
by `(1,6)` exactly once. The **last historical frame's** wrists are stacked in
left/right order as `[1,2,3,240,320]`, float RGB in `[0,1]`, and supplied through
`infer_stage2_window(..., wrist_images=...)`. The adapter performs no wrist resize,
padding or ImageNet normalization. The upstream shared frozen DINOv2 encoder performs
antialiased bicubic resize to 256×192, reflect padding to 266×196 and ImageNet
normalization once, producing 266 patch tokens per wrist. Sampling iterations reuse
these encoded conditions.

Required third-view metadata: runtime intrinsics, world-to-OpenCV-camera extrinsics,
`additional_info.third_camera_source=observer_camera`, `T_world_from_robot_base`,
and consecutive `waypoint_index` values. Measured six-joint arm states come from
`state.left_arm_real_joint_state` and `state.right_arm_real_joint_state`, in radians.
Drive-target joint fields do not initialize queries.

For every window, measured-joint FK from its **first frame** produces the fingertip
TCPs used to initialize the two image queries, ordered left/right regardless of
image order. Queries outside the image are clipped as in training; non-finite or
behind-camera queries fail. Subsequent historical projections are reconstructed by
4RC, with no GT-history conditioning or propagated-query fallback.

## History, actions and control

Bootstrap repeats the first observation K times. After each executed waypoint the
evaluator sends one observation, including the terminal observation. The next window
retains the latest K frames; no boundary observation is duplicated. With the default
K=4, N=8, windows use `[0,0,0,0]`, then `[5,6,7,8]`, then `[13,14,15,16]`.
Their wrist inputs come from waypoint 0, 8 and 16 respectively. Reset clears all history.

The model predicts H ordered, absolute TCP poses in the last historical camera frame.
It conditions on step indices, not timestamps or an execution frequency.
`--frequency 15` is shared evaluator metadata; each waypoint's actual duration varies
with the planner trajectory. Collection `save_freq: 15` is a physics-step stride,
not a 15 Hz control rate.

The upstream model quantile-unnormalizes the full 20-dimensional action vector
before decoding xyz, rotation6d and gripper score. The adapter receives metric TCP
positions and rotation matrices, so it never applies another unnormalize.
Gripper score ≥0 means open; returned grippers are binary 1=open, 0=closed.

Coordinate conversion uses the last historical camera and actual robot-root pose:

```text
T_base_from_tcp = inverse(T_world_from_base) @ inverse(T_camera_from_world) @ T_camera_from_tcp
p_ee_base = p_tcp_base - R_tcp_base @ fingertip_offset
T_world_from_ee = T_world_from_base @ T_base_from_ee
```

The fingertip contact midpoint is the training TCP. Its offset to the control EE
is computed from the URDF; the validated assets give approximately
`[0.147335,-0.0000015,-0.000713515]` metres. Both position and orientation are
transformed. Actions use `left_ee_pose` / `right_ee_pose` as `[x,y,z,qw,qx,qy,qz]`
and scalar-array `left_ee_joint_state` / `right_ee_joint_state` for grippers.

The evaluator submits the first N actions **one waypoint at a time** to
`take_action(..., action_type="ee")`. RoboTwin plans each arm from its current
measured joints, executes the trajectory, and then plans the next waypoint.
The existing per-arm planning-failure handling remains: skip the failed arm's
movement while gripper control and the other arm continue. Invalid model outputs
fail the episode; pre-action simulator failures retain the evaluator's retry behavior.

## Visualization and outputs

`--viser` starts the manual UI on the policy host, normally at
http://127.0.0.1:8020. The actual URL is printed. One click predicts H steps,
updates the scene, executes the first N and pauses again. History browsing uses
K frames and future browsing uses H steps. `--viser-auto` enables automatic inference.
Manual UI failures stop evaluation rather than bypassing click gating.
`--viser-host` and `--viser-port` configure the listener and also enable the viewer.

The viewer retains third-view RGB with TCP overlays, reconstructed historical
point clouds, predicted fingertip TCPs and measured execution in world coordinates.
Point clouds use predicted metric depth, with aligned unpadding and runtime calibration.
The display does not feed geometry back into the policy. Default point sampling keeps
one in every two valid points, capped at 100000 per cloud; up to 128 windows are retained.
Viewer history is held in memory, and closes when the local policy server stops.

Results and videos share:

```text
eval_result/rollout/<task>/<policy>/<task_config>/<checkpoint>/<timestamp>/
  _result.txt
  episode0.mp4
  episode0_third_view.mp4
```

No per-waypoint images or NPZ files are written. Prediction JSONL records under
`eval_result/FourRC_predictions/` contain all H base-frame TCP poses, grippers,
K/H/N, historical waypoint indices, wrist anchor index, queries, calibration,
inference duration and trial outcomes. `FOURRC_LOG_DIR` overrides this directory.

## Remote serving

On the policy host, from its RoboTwin root:

```bash
FOURRC_EXECUTE_HORIZON=8 bash XPolicyLab/policy/FourRC/setup_eval_policy_server.sh \
  RoboTwin adjust_bottle /absolute/path/to/wrist/checkpoint arx_x5 ee 0 0 4rc 18080 0.0.0.0
```

On the simulator host, keep checkpoint metadata available locally and connect:

```bash
bash scripts/eval_4rc.sh --task adjust_bottle --test-num 2 -- \
  --enable-remote --policy-server-ip SERVER_IP --policy-server-port 18080
```

A remote server needs the adapter, upstream 4RC, complete checkpoint, T5 resources
and RoboTwin robot assets. Configure execution length and visualization when starting
that server; client-side flags cannot reconfigure an already running server.
For a loopback viewer, use `ssh -L 8020:127.0.0.1:8020 POLICY_HOST`.
Do not interleave simultaneous environments on one stateful server.

## Checks

```bash
conda run -n 4rc python -m pytest XPolicyLab/policy/FourRC/tests -q
bash scripts/eval_4rc.sh --dry-run
```

Tests cover wrist frame selection and RGB ranges, variable K/H/N, quantile checkpoint
requirements, strict model loading boundaries, measured-joint query initialization,
coordinate transforms, three-camera WebSocket transport, terminal observations,
viewer ranges and manual gating. The generic demo debug client lacks the calibrated
third-camera/root metadata: use the calibrated transport tests rather than relaxing
the adapter contract. Real model/simulator smoke testing requires complete weights
and a usable GPU; dry-run and fake-policy tests do not establish task success rates.
