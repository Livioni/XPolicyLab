"""Live, bounded-memory viser display of the geometry used by each policy window."""
from collections import deque
import os
import threading

import numpy as np
import cv2
from scipy.spatial.transform import Rotation


ARM_COLORS = ((255, 145, 40), (70, 160, 255))
FK_COLORS = ((255, 225, 60), (80, 230, 130))


def project_tcp_pixels(points_world, camera_from_world, intrinsics, image_shape):
    """Project world TCP centres into the original, unpadded RGB image."""
    points = np.asarray(points_world, dtype=np.float64)
    camera = points @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
    pixels = camera @ intrinsics.T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = pixels[..., :2] / pixels[..., 2:3]
    height, width = image_shape[:2]
    valid = (np.isfinite(uv).all(-1) & np.isfinite(camera).all(-1) & (camera[..., 2] > 1e-6)
             & (uv[..., 0] >= 0) & (uv[..., 0] < width)
             & (uv[..., 1] >= 0) & (uv[..., 1] < height))
    return uv, valid


def tcp_overlay(rgb, intrinsics, camera_from_world, *, measured=None, future=None,
                recovered=None, future_step=1):
    """Draw RGB annotations on a copy; policy inputs are never changed."""
    image = np.array(rgb, copy=True)
    for points, colors, name, radius in (
            (measured, FK_COLORS, "FK", 5),
            (recovered, ARM_COLORS, "REC", 3),
            (future, ARM_COLORS, f"+{future_step}", 8)):
        if points is None:
            continue
        uv, valid = project_tcp_pixels(points, camera_from_world, intrinsics, image.shape)
        for arm, color in enumerate(colors):
            if not valid[arm]:
                continue
            x, y = np.rint(uv[arm]).astype(int)
            x, y = min(x, image.shape[1] - 1), min(y, image.shape[0] - 1)
            cv2.circle(image, (x, y), radius + 1, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.circle(image, (x, y), radius, color, 2, cv2.LINE_AA)
            cv2.drawMarker(image, (x, y), color, cv2.MARKER_CROSS, 5, 1, cv2.LINE_AA)
            label = f"{'L' if arm == 0 else 'R'} {name}"
            (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .35, 1)
            origin = (max(0, min(x + 10, image.shape[1] - width - 1)),
                      max(height + 2, min(y - 7 if name == "FK" else y + 15, image.shape[0] - 4)))
            cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1, cv2.LINE_AA)
    return image


def scene_record(frames, prediction, base_tcp, make_point_cloud, *, max_points,
                 case, instruction, window, execute_horizon, downsample=2):
    """Convert everything to simulator WORLD; future poses reuse control geometry.

    Reconstruction depth includes reflect padding. Remove it before using the
    original RGB/intrinsics. Historical TCPs each belong to their own camera.
    Downsample counts filtered points: 2 retains every second valid point.
    """
    history_frames = len(frames)
    anchor = frames[-1]
    depth = np.asarray(prediction["depth"])
    confidence = np.asarray(prediction["depth_confidence"])
    if depth.shape != (history_frames, 252, 322) or confidence.shape != depth.shape:
        raise ValueError(f"Viser expects 4RC depth/confidence [{history_frames},252,322]")
    if isinstance(downsample, bool) or not isinstance(downsample, int) or downsample < 1:
        raise ValueError("Viser downsample must be a positive integer")
    camera_to_world = np.linalg.inv(np.stack([frame["cw"] for frame in frames]))
    clouds = []
    for local, frame in enumerate(frames):
        # Project at native resolution, then sample valid points so a factor of
        # two halves the point count while preserving the original camera rays.
        points, colors = make_point_cloud(
            depth[local, 6:246, 1:321],
            confidence[local, 6:246, 1:321],
            frame["rgb"], frame["k"],
            frame["cw"], np.eye(4), confidence_percentile=2.5,
            max_points=0, max_depth=3.0, seed=window * history_frames + local)
        points, colors = points[::downsample], colors[::downsample]
        if max_points and len(points) > max_points:
            indices = np.random.default_rng(window * history_frames + local).choice(len(points), max_points, replace=False)
            points, colors = points[indices], colors[indices]
        points, colors = points.copy(), colors.copy()
        clouds.append(dict(points=points, colors=colors, rgb=frame["rgb"],
                           index=frame["index"], camera_to_world=camera_to_world[local],
                           intrinsics=frame["k"].copy(), camera_from_world=frame["cw"].copy()))
    recovered = np.einsum("tij,taj->tai", camera_to_world[:, :3, :3],
                          prediction["history_position"]) + camera_to_world[:, None, :3, 3]
    valid = np.asarray(prediction["history_valid"], dtype=bool)
    if valid.shape != (history_frames, 2):
        raise ValueError(f"Viser expects history_valid [{history_frames},2]")
    recovered[~valid] = np.nan
    measured = np.stack([frame["wb"] @ frame["tcp_base"] for frame in frames])
    return dict(case=dict(case), instruction=instruction, window=window,
                anchor=anchor["index"], execute_horizon=execute_horizon,
                clouds=clouds, downsample=downsample,
                world_from_base=anchor["wb"].copy(),
                future=(anchor["wb"] @ base_tcp).astype(np.float32),
                gripper=np.asarray(prediction["action_gripper"]).copy(),
                history_indices=[frame["index"] for frame in frames],
                recovered=recovered.astype(np.float32), measured=measured.astype(np.float32),
                executed=[measured[-1].astype(np.float32)], executed_indices=[anchor["index"]],
                result=None)


def _boolean(value, name):
    if isinstance(value, str):
        if value.lower() not in ("true", "false", "1", "0"):
            raise ValueError(f"{name} must be true/false or 1/0")
        return value.lower() in ("true", "1")
    return bool(value)


def create_viewer(config, make_point_cloud, *, history_frames, prediction_horizon):
    if not _boolean(os.environ.get("FOURRC_VISER", config.get("viser", False)), "FOURRC_VISER"):
        return None
    return LiveViewer(
        make_point_cloud, history_frames=history_frames, prediction_horizon=prediction_horizon,
        host=os.environ.get("FOURRC_VISER_HOST", config.get("viser_host", "127.0.0.1")),
        port=int(os.environ.get("FOURRC_VISER_PORT", config.get("viser_port", 8020))),
        max_points=int(config.get("viser_max_points", 100000)),
        max_windows=int(config.get("viser_max_windows", 128)),
        manual=_boolean(os.environ.get("FOURRC_VISER_MANUAL", config.get("viser_manual", True)), "FOURRC_VISER_MANUAL"),
        downsample=int(config.get("viser_downsample", 2)),
        point_size=float(config.get("viser_point_size", 0.003)))


class LiveViewer:
    def __init__(self, make_point_cloud, *, history_frames, prediction_horizon, host="127.0.0.1", port=8020,
                 max_points=100000, max_windows=128, manual=True, downsample=2, point_size=0.003):
        if (not host or not 1 <= port <= 65535 or max_points < 1 or max_windows < 1
                or isinstance(downsample, bool) or not isinstance(downsample, int) or downsample < 1
                or not np.isfinite(point_size) or not 0.001 <= point_size <= 0.012):
            raise ValueError("Invalid viser host/port, limits, downsample or point size")
        try:
            import viser
        except ImportError as exc:
            raise RuntimeError("Install viser in the policy environment: pip install 'viser>=1.1,<2'") from exc
        self.history_frames = history_frames
        self.prediction_horizon = prediction_horizon
        self.make_point_cloud = make_point_cloud
        self.max_points = max_points
        self.manual = manual
        self.downsample = downsample
        self._phase = "idle"
        self.records = deque(maxlen=max_windows)
        self._lock = threading.RLock()
        self._syncing = False
        self._closed = False
        self._nodes, self._actual_nodes = [], []
        self.server = viser.ViserServer(host=host, port=port, label="4RC · RoboTwin live inference")
        self.server.scene.set_up_direction("+z")
        self.server.scene.add_frame("/world", axes_length=0.15, axes_radius=0.003)
        gui = self.server.gui
        with gui.add_folder("推理控制"):
            self.run_button = gui.add_button("推理并执行一次", disabled=True, color="blue")
            self.control_status = gui.add_markdown("等待环境准备…" if manual else "自动推理模式")
            gui.add_markdown(f"每次点击：推理 {prediction_horizon} 步 → 更新场景 → 执行前 N 步 → 等待下次点击。")
        with gui.add_folder("历史窗口与帧"):
            self.follow = gui.add_checkbox("Follow latest", True)
            self.window = gui.add_slider("Window", min=0, max=1, step=1, initial_value=0, disabled=True)
            self.history_frame = gui.add_slider(f"历史帧 (0–{history_frames - 1})", min=0, max=max(1, history_frames - 1), step=1,
                                                initial_value=history_frames - 1, disabled=True)
            with gui.add_folder("逐帧浏览", expand_by_default=True):
                previous_frame = gui.add_button("上一帧")
                next_frame = gui.add_button("下一帧")
            self.step = gui.add_slider("Future step", min=1, max=max(2, prediction_horizon), step=1,
                                       initial_value=prediction_horizon, disabled=prediction_horizon == 1)
            camera_button = gui.add_button("Third-camera view")
        with gui.add_folder("Geometry / TCP"):
            self.cloud = gui.add_checkbox("Reconstructed point cloud", True)
            self.future = gui.add_checkbox("Predicted future TCP", True)
            self.measured = gui.add_checkbox("Measured FK / execution", True)
            self.recovered = gui.add_checkbox("Recovered history TCP", False)
            self.axes = gui.add_checkbox("TCP orientation axes", True)
            self.point_size = gui.add_slider("Point size (m)", min=0.001, max=0.012,
                                            step=0.001, initial_value=point_size)
        gui.add_markdown("**Future:** left orange / right blue. Thick lines: the first N commands.\n\n"
                         "**Measured FK:** left yellow / right green.\n\n"
                         "Point cloud: 4RC predicted metric depth. All geometry: simulator world, metres.")
        self.info = gui.add_markdown("Waiting for the first inference window…")
        self.rgb_panel = gui.add_panel()
        with self.rgb_panel.add_tab("第三人称 RGB / TCP"):
            self.rgb = gui.add_image(np.zeros((240, 320, 3), dtype=np.uint8),
                                     label="Selected historical third view", format="png")
            gui.add_markdown("**TCP 投影**：L 左臂 / R 右臂。\n\n"
                             "**FK**：当前所选历史帧，左黄 / 右绿。\n\n"
                             "**+N**：Future step 选中的预测点，左橙 / 右蓝；投影到所选 RGB。\n\n"
                             "REC 为恢复的历史 TCP，随对应开关显示。图外或相机背后的点不标注。")
        self.rgb_panel.dock_left()
        self.rgb_panel.set_width(380)
        self.run_button.on_click(self._request_inference)
        self.history_frame.on_update(lambda _: self.refresh() if not self._syncing else None)
        previous_frame.on_click(lambda _: self._step_history(-1))
        next_frame.on_click(lambda _: self._step_history(1))
        self.window.on_update(self._select_window)
        self.follow.on_update(self._follow_latest)
        for control in (self.step, self.cloud, self.future, self.measured,
                        self.recovered, self.axes, self.point_size):
            control.on_update(lambda _: self.refresh())
        camera_button.on_click(lambda _: self.reset_camera())
        self.server.on_client_connect(lambda client: self.reset_camera(client))
        display_host = "127.0.0.1" if host == "0.0.0.0" else host
        self.url = f"http://{display_host}:{self.server.get_port()}"
        print(f"[FourRC viser] {self.url} (policy-server host; live during evaluation)", flush=True)

    def inference_ready(self):
        """Short RPC polling arms the button; never hold a policy RPC open."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Manual inference viewer has been closed")
            if not self.manual:
                return True
            if self._phase == "idle":
                self._phase = "waiting"
                self.run_button.disabled = False
                self.control_status.content = "**等待点击「推理并执行一次」**。可先浏览历史帧。"
            return self._phase == "requested"

    def _request_inference(self, _=None):
        with self._lock:
            if self._closed or not self.manual or self._phase != "waiting":
                return
            self._phase = "requested"
            self.run_button.disabled = True
            self.follow.value = True
            self.control_status.content = "**已触发一次推理**，正在等待计算结果…"

    def begin_inference(self):
        with self._lock:
            if self._closed or (self.manual and self._phase != "requested"):
                raise RuntimeError("Click the inference button before requesting an action")
            self._phase = "inference"
            self.run_button.disabled = True
            self.control_status.content = "**推理中…**"

    def _step_history(self, direction):
        with self._lock:
            if self.records:
                self.follow.value = False
                self.history_frame.value = max(0, min(self.history_frames - 1, self.history_frame.value + direction))

    def _selected(self):
        return next((record for record in self.records if record["window"] == self.window.value), None)

    def reset(self):
        with self._lock, self.server.atomic():
            self.records.clear()
            self._phase = "idle"
            self.run_button.disabled = True
            self.control_status.content = "等待环境准备…" if self.manual else "自动推理模式"
            self._clear(self._nodes)
            self._clear(self._actual_nodes)
            self._syncing = True
            try:
                self.window.min, self.window.max, self.window.value = 0, 1, 0
                self.window.disabled = True
                self.history_frame.value = self.history_frames - 1
                self.history_frame.disabled = True
                self.follow.value = True
            finally:
                self._syncing = False
            self.info.content = "Waiting for the first inference window…"
            self.rgb.image = np.zeros((240, 320, 3), dtype=np.uint8)

    def publish(self, frames, prediction, base_tcp, *, case, instruction, window, execute_horizon):
        record = scene_record(frames, prediction, base_tcp, self.make_point_cloud,
                              max_points=self.max_points, case=case, instruction=instruction,
                              window=window, execute_horizon=execute_horizon, downsample=self.downsample)
        with self._lock:
            first = not self.records
            self.records.append(record)
            self._phase = "executing"
            self.control_status.content = f"**推理完成，执行前 {execute_horizon} 步…**"
            self.history_frame.disabled = self.history_frames == 1
            self._syncing = True
            try:
                self.window.max = max(1, record["window"])
                self.window.min = self.records[0]["window"]
                self.window.disabled = len(self.records) == 1
                if self.follow.value or self.window.value < self.window.min:
                    self.window.value = record["window"] if self.follow.value else self.window.min
                    self.history_frame.value = self.history_frames - 1
            finally:
                self._syncing = False
            self.refresh()
            if first:
                self.reset_camera()

    def observe(self, frame):
        with self._lock:
            if not self.records:
                self.rgb.image = tcp_overlay(frame["rgb"], frame["k"], frame["cw"],
                                             measured=(frame["wb"] @ frame["tcp_base"])[:, :3, 3])
                return
            record = self.records[-1]
            index = frame["index"]
            if not record["anchor"] < index <= record["anchor"] + record["execute_horizon"]:
                return
            if index <= record["executed_indices"][-1]:
                return
            record["executed"].append((frame["wb"] @ frame["tcp_base"]).astype(np.float32))
            record["executed_indices"].append(index)
            if frame["terminal"]:
                self._phase = "finished"
                self.run_button.disabled = True
                self.control_status.content = "**本次评测结束。**"
            elif len(record["executed"]) - 1 == record["execute_horizon"]:
                self._phase = "idle"
                self.control_status.content = "**本段执行完成**，准备下一次推理。"
            if self._selected() is record:
                with self.server.atomic():
                    self._draw_actual(record)
                    self._update_info(record)

    def finish(self, result):
        with self._lock:
            self._phase = "finished"
            self.run_button.disabled = True
            self.control_status.content = "**本次评测结束。**"
            for record in self.records:
                record["result"] = dict(result or {})
            if self.records:
                self.refresh()

    def _select_window(self, event):
        if self._syncing:
            return
        with self._lock:
            if event.client is not None:
                self.follow.value = False
            self._syncing = True
            try:
                self.history_frame.value = self.history_frames - 1
            finally:
                self._syncing = False
            self.refresh()

    def _follow_latest(self, _):
        if self._syncing:
            return
        with self._lock:
            if self.follow.value and self.records:
                self.window.value = self.records[-1]["window"]
            self.refresh()

    @staticmethod
    def _clear(nodes):
        for node in nodes:
            node.remove()
        nodes.clear()

    def _trajectory(self, name, positions, colors, width, nodes, *, point_size=0.007):
        for arm, color in enumerate(colors):
            p = np.asarray(positions[:, arm], dtype=np.float32)
            valid = np.isfinite(p).all(-1)
            if valid.any():
                nodes.append(self.server.scene.add_point_cloud(
                    f"/{name}/{arm}/points", points=p[valid], colors=color,
                    point_size=point_size, point_shape="circle", precision="float32"))
            if len(p) > 1:
                segments = np.stack((p[:-1], p[1:]), axis=1)[valid[:-1] & valid[1:]]
                if len(segments):
                    nodes.append(self.server.scene.add_line_segments(
                        f"/{name}/{arm}/lines", points=segments, colors=color,
                        thickness=width, thickness_units="screen"))

    def _frame(self, name, pose, nodes, *, length=0.04):
        if np.isfinite(pose).all():
            xyzw = Rotation.from_matrix(pose[:3, :3]).as_quat()
            nodes.append(self.server.scene.add_frame(name, position=pose[:3, 3],
                wxyz=xyzw[[3, 0, 1, 2]], axes_length=length, axes_radius=0.0015))

    def _draw_actual(self, record):
        self._clear(self._actual_nodes)
        if self.measured.value:
            poses = np.stack(record["executed"])
            self._trajectory("executed", poses[..., :3, 3], FK_COLORS, 4, self._actual_nodes)
            if self.axes.value:
                for arm in range(2):
                    self._frame(f"/actual_axes/{arm}", poses[-1, arm], self._actual_nodes)

    def _update_info(self, record):
        step = min(int(self.step.value), self.prediction_horizon)
        case = record["case"]
        grip = record["gripper"][step - 1]
        selected_cloud = record["clouds"][min(int(self.history_frame.value), self.history_frames - 1)]
        status = "running" if record["result"] is None else ("success" if record["result"].get("success") else "failed")
        self.info.content = (
            f"**{case.get('task_name', 'episode')} · seed {case.get('seed', '?')} · {status}**\n\n"
            f"{record['instruction']}\n\nWindow **{record['window']}**, anchor waypoint **{record['anchor']}**; "
            f"history {record['history_indices']}.\n\n"
            f"显示点云：历史位置 **{self.history_frame.value}** / waypoint "
            f"**{selected_cloud['index']}**；"
            f"有效点按 **1/{record['downsample']}** 抽样，当前 **{len(selected_cloud['points']):,}** 点。"
            f"未来轨迹属于窗口末帧。\n\n"
            f"Predict **{len(record['future'])}**, execute **{record['execute_horizon']}**; "
            f"received **{len(record['executed']) - 1}** post-action observations.\n\n"
            f"Selected future **+{step}** ({'commanded prefix' if step <= record['execute_horizon'] else 'not executed in this chunk'}). "
            f"Gripper: left {'open' if grip[0] else 'closed'}, right {'open' if grip[1] else 'closed'}.\n\n"
            f"Retaining {len(self.records)}/{self.records.maxlen} windows in memory.")

    def refresh(self):
        with self._lock, self.server.atomic():
            record = self._selected()
            if record is None or self._closed:
                return
            self._clear(self._nodes)
            selected_cloud = record["clouds"][min(int(self.history_frame.value), self.history_frames - 1)]
            local = min(int(self.history_frame.value), self.history_frames - 1)
            self.rgb.image = tcp_overlay(
                selected_cloud["rgb"], selected_cloud["intrinsics"], selected_cloud["camera_from_world"],
                measured=record["measured"][local, :, :3, 3] if self.measured.value else None,
                future=record["future"][min(int(self.step.value), self.prediction_horizon) - 1, :, :3, 3] if self.future.value else None,
                recovered=record["recovered"][local] if self.recovered.value else None,
                future_step=min(int(self.step.value), self.prediction_horizon))
            if self.cloud.value and len(selected_cloud["points"]):
                self._nodes.append(self.server.scene.add_point_cloud("/cloud", points=selected_cloud["points"],
                    colors=selected_cloud["colors"], point_size=float(self.point_size.value),
                    point_shape="rounded", precision="float32"))
            self._frame("/robot_base", record["world_from_base"], self._nodes, length=0.12)
            self._frame("/third_camera", selected_cloud["camera_to_world"], self._nodes, length=0.08)
            if self.recovered.value:
                self._trajectory("recovered", record["recovered"], ARM_COLORS, 1, self._nodes, point_size=0.004)
            if self.measured.value:
                self._trajectory("history_fk", record["measured"][..., :3, 3], FK_COLORS, 1, self._nodes, point_size=0.004)
            if self.future.value:
                positions = np.concatenate((record["measured"][-1:, :, :3, 3], record["future"][..., :3, 3]))
                self._trajectory("future", positions, ARM_COLORS, 2, self._nodes, point_size=0.005)
                self._trajectory("commanded", positions[:record["execute_horizon"] + 1], ARM_COLORS, 6, self._nodes)
                selected = record["future"][min(int(self.step.value), self.prediction_horizon) - 1]
                self._trajectory("selected", selected[None, :, :3, 3], ARM_COLORS, 1, self._nodes, point_size=0.015)
                if self.axes.value:
                    for arm in range(2):
                        self._frame(f"/future_axes/{arm}", selected[arm], self._nodes)
            self._draw_actual(record)
            self._update_info(record)

    def reset_camera(self, client=None):
        with self._lock:
            record = self._selected()
            if record is None:
                return
            pose = record["clouds"][min(int(self.history_frame.value), self.history_frames - 1)]["camera_to_world"]
            clients = [client] if client is not None else list(self.server.get_clients().values())
            for target in clients:
                target.camera.position = pose[:3, 3] - 0.2 * pose[:3, 2]
                target.camera.look_at = pose[:3, 3] + 1.0 * pose[:3, 2]
                target.camera.up_direction = -pose[:3, 1]

    def close(self):
        self._closed = True
        self.server.stop()
