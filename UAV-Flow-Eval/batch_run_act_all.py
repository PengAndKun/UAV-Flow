import base64
import requests
import sys
sys.path.append('./')
from io import BytesIO
import json
import numpy as np
from PIL import Image
import gym
import gym_unrealcv
import time
import argparse
import os
import cv2
import re
from relative import calculate_new_pose
glob = __import__('glob')
from gym_unrealcv.envs.wrappers import time_dilation, configUE, augmentation
from gym_unrealcv.envs.utils import misc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
from typing import Any, Dict, List, Optional, Tuple
from matplotlib.ticker import MultipleLocator
logger = logging.getLogger(__name__)

# ====== Constants ======
IMG_INPUT_SIZE: Tuple[int, int] = (224, 224)
SLEEP_SHORT_S: float = 1.0
SLEEP_AFTER_RESET_S: float = 2.0
ACTION_SMALL_DELTA_POS: float = 3.0
ACTION_SMALL_DELTA_YAW: float = 1.0
ACTION_SMALL_STEPS: int = 10
DEBUG_IMAGE_PATH: str = './debug.jpg'
TRAJ_IMG_SIZE_2D: Tuple[int, int] = (10, 10)
TRAJ_IMG_SIZE_3D: Tuple[int, int] = (12, 10)
PLOT_YAW_ARROW_MIN_LEN_2D: int = 10
PLOT_YAW_ARROW_MIN_LEN_3D: int = 12
UAV_PREVIEW_WINDOW_NAME: str = "UAV Preview"
UAV_PREVIEW_WINDOW_OPENED: bool = False



def send_prediction_request(
    image: Image,
    proprio: np.ndarray,
    instr: str,
    server_url: str,
    target_local: Optional[List[float]] = None,
) -> Optional[Dict[str, Any]]:
    """Send a request to the inference service and return JSON response.

    Args:
        image: PIL image object, resized to 224x224 and sent as PNG.
        proprio: Vehicle state vector (np.ndarray), converted to list.
        instr: Text instruction.
        server_url: Inference service /predict endpoint URL.
    Returns:
        dict or None: Parsed JSON if successful, otherwise None on error.
    """
    proprio_list = proprio.tolist()
    img_io = BytesIO()
    if image.size != IMG_INPUT_SIZE:
        image = image.resize(IMG_INPUT_SIZE)
    image.save(img_io, format='PNG')
    img_data = img_io.getvalue()
    img_base64 = base64.b64encode(img_data).decode('utf-8')
    payload: Dict[str, Any] = {
        'image': img_base64,
        'proprio': proprio_list,
        'instr': instr
    }
    if target_local is not None:
        payload['target_local'] = [float(v) for v in target_local[:3]]
    headers = {
        'Content-Type': 'application/json'
    }
    try:
        response = requests.post(
            server_url,
            data=json.dumps(payload),
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return None


def draw_2d_trajectory_from_log(log_path: str, save_path: str, instruction: str, target: Optional[List[float]], tick_x: Optional[float] = None, tick_y: Optional[float] = None, min_span_x: Optional[float] = None, min_span_y: Optional[float] = None) -> None:
    """Plot 2D trajectory from log and save the image.

    - Fixed tick step: use tick_x/tick_y to specify grid spacing
    - Minimum span: use min_span_x/min_span_y to expand view when data span is smaller
    """
    with open(log_path, 'r') as f:
        log = json.load(f)
    if not isinstance(log, list) or len(log) == 0:
        plt.figure(figsize=TRAJ_IMG_SIZE_2D)
        plt.title(instruction + '\n' + log_path + ' (no data)')
        plt.xlabel("Y (right)")
        plt.ylabel("X (forward)")
        ax = plt.gca()
        if tick_y is not None and tick_y > 0:
            ax.xaxis.set_major_locator(MultipleLocator(tick_y))
        if tick_x is not None and tick_x > 0:
            ax.yaxis.set_major_locator(MultipleLocator(tick_x))
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        return
    trajectory = np.array([item['state'][0] + item['state'][1] for item in log])
    if trajectory.ndim != 2 or trajectory.shape[1] < 5:
        plt.figure(figsize=TRAJ_IMG_SIZE_2D)
        plt.title(instruction + '\n' + log_path + ' (invalid data)')
        plt.xlabel("Y (right)")
        plt.ylabel("X (forward)")
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        return
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    yaw = np.deg2rad(trajectory[:, 4])
    arrow_length = max(PLOT_YAW_ARROW_MIN_LEN_2D, np.sqrt(x.var() + y.var()) // 4)
    dx = np.cos(yaw) * arrow_length
    dy = np.sin(yaw) * arrow_length

    plt.figure(figsize=TRAJ_IMG_SIZE_2D)
    if target is not None:
        plt.scatter(target[1], target[0], color='red', label='target')
    plt.plot(y, x, color='blue', label='trajectory')
    plt.quiver(y, x, dy, dx, angles='xy', scale_units='xy', scale=1, color='green', width=0.003, label='yaw')

    ax = plt.gca()
    # Set equal aspect ratio first, then expand to minimum span based on this
    ax.set_aspect('equal', adjustable='box')
    
    # Expand to minimum span without shrinking: X-axis shows Y, Y-axis shows X
    cur_xmin, cur_xmax = ax.get_xlim()
    cur_ymin, cur_ymax = ax.get_ylim()
    span_y_axis = cur_xmax - cur_xmin
    span_x_axis = cur_ymax - cur_ymin
    if min_span_y is not None and min_span_y > 0 and span_y_axis < min_span_y:
        cx = (cur_xmin + cur_xmax) / 2.0
        half = min_span_y / 2.0
        ax.set_xlim(cx - half, cx + half)
    if min_span_x is not None and min_span_x > 0 and span_x_axis < min_span_x:
        cy = (cur_ymin + cur_ymax) / 2.0
        half = min_span_x / 2.0
        ax.set_ylim(cy - half, cy + half)

    # Fixed tick spacing (grid size)
    if tick_y is not None and tick_y > 0:
        ax.xaxis.set_major_locator(MultipleLocator(tick_y))
    if tick_x is not None and tick_x > 0:
        ax.yaxis.set_major_locator(MultipleLocator(tick_x))

    plt.legend()
    plt.title(instruction + '\n' + log_path)
    plt.xlabel("Y (right)")
    plt.ylabel("X (forward)")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def draw_3d_trajectory_from_log(log_path: str, save_path: str, instruction: str, target: Optional[List[float]], tick_x: Optional[float] = None, tick_y: Optional[float] = None, tick_z: Optional[float] = None, min_span_x: Optional[float] = None, min_span_y: Optional[float] = None, min_span_z: Optional[float] = None) -> None:
    """Plot 3D trajectory from log and save the image.

    - Fixed tick step: use tick_x/tick_y/tick_z to specify grid spacing
    - Minimum span: use min_span_x/min_span_y/min_span_z to expand view when data span is smaller
    """
    with open(log_path, 'r') as f:
        log = json.load(f)
    if not isinstance(log, list) or len(log) == 0:
        fig = plt.figure(figsize=TRAJ_IMG_SIZE_3D)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"3D Trajectory with Yaw Direction\nInstruction: {instruction} (no data)")
        ax.set_xlabel("Y (right)")
        ax.set_ylabel("X (forward)")
        ax.set_zlabel("Z (up)")
        if tick_y is not None and tick_y > 0:
            ax.xaxis.set_major_locator(MultipleLocator(tick_y))
        if tick_x is not None and tick_x > 0:
            ax.yaxis.set_major_locator(MultipleLocator(tick_x))
        if tick_z is not None and tick_z > 0:
            ax.zaxis.set_major_locator(MultipleLocator(tick_z))
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return
    trajectory = np.array([item['state'][0] + item['state'][1] for item in log])
    if trajectory.ndim != 2 or trajectory.shape[1] < 5:
        fig = plt.figure(figsize=TRAJ_IMG_SIZE_3D)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"3D Trajectory with Yaw Direction\nInstruction: {instruction} (invalid data)")
        ax.set_xlabel("Y (right)")
        ax.set_ylabel("X (forward)")
        ax.set_zlabel("Z (up)")
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]
    yaw = np.deg2rad(trajectory[:, 4])
    arrow_length =  max(PLOT_YAW_ARROW_MIN_LEN_3D, np.sqrt(x.var() + y.var()) // 4)
    dx = np.cos(yaw) * arrow_length
    dy = np.sin(yaw) * arrow_length
    dz = np.zeros_like(dx)

    fig = plt.figure(figsize=TRAJ_IMG_SIZE_3D)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(y, x, z, color='blue', label='Trajectory')
    ax.quiver(y, x, z, dy, dx, dz, color='green', length=1.0, normalize=False, linewidth=0.5, label='Yaw direction')
    ax.scatter(y[0], x[0], z[0], color='blue', s=50)
    ax.text(y[0], x[0], z[0], 'Start', color='blue', fontsize=12)
    if target is not None:
        ax.scatter(target[1], target[0], target[2], color='red', s=50)
        ax.text(target[1], target[0], target[2], 'Target', color='red', fontsize=12)

    # Expand to minimum span without shrinking. Axis mapping: x-axis shows Y, y-axis shows X
    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()
    cur_zlim = ax.get_zlim()
    span_y = cur_xlim[1] - cur_xlim[0]
    span_x = cur_ylim[1] - cur_ylim[0]
    span_z = cur_zlim[1] - cur_zlim[0]
    if min_span_y is not None and min_span_y > 0 and span_y < min_span_y:
        c = (cur_xlim[0] + cur_xlim[1]) / 2.0
        half = min_span_y / 2.0
        ax.set_xlim(c - half, c + half)
    if min_span_x is not None and min_span_x > 0 and span_x < min_span_x:
        c = (cur_ylim[0] + cur_ylim[1]) / 2.0
        half = min_span_x / 2.0
        ax.set_ylim(c - half, c + half)
    if min_span_z is not None and min_span_z > 0 and span_z < min_span_z:
        c = (cur_zlim[0] + cur_zlim[1]) / 2.0
        half = min_span_z / 2.0
        ax.set_zlim(c - half, c + half)

    # Fixed tick spacing
    if tick_y is not None and tick_y > 0:
        ax.xaxis.set_major_locator(MultipleLocator(tick_y))
    if tick_x is not None and tick_x > 0:
        ax.yaxis.set_major_locator(MultipleLocator(tick_x))
    if tick_z is not None and tick_z > 0:
        ax.zaxis.set_major_locator(MultipleLocator(tick_z))

    # Equal aspect box: use current axis limits to avoid flattening when Z is zero
    _xl = ax.get_xlim(); _yl = ax.get_ylim(); _zl = ax.get_zlim()
    ax.set_box_aspect(((max(_xl[1]-_xl[0], 1e-6)), (max(_yl[1]-_yl[0], 1e-6)), (max(_zl[1]-_zl[0], 1e-6))))
    ax.set_title(f"3D Trajectory with Yaw Direction\nInstruction: {instruction}")
    ax.set_xlabel("Y (right)")
    ax.set_ylabel("X (forward)")
    ax.set_zlabel("Z (up)")
    ax.legend()
    ax.grid(True)
    ax.view_init(elev=30, azim=-100)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def get_policy_cam_id(env: Any) -> int:
    """Get the camera id used for UAV observation/policy."""
    cam_list = getattr(env.unwrapped, "cam_list", None)
    if isinstance(cam_list, list) and len(cam_list) > 0:
        return int(cam_list[0])
    return 0


def compute_look_at_rotation(
    camera_pos: Tuple[float, float, float],
    target_pos: Tuple[float, float, float],
    rotation_offset: Tuple[float, float, float],
) -> List[float]:
    """Compute a camera rotation that looks at the target with optional offsets."""
    cam_x, cam_y, cam_z = camera_pos
    target_x, target_y, target_z = target_pos
    delta_x = target_x - cam_x
    delta_y = target_y - cam_y
    delta_z = target_z - cam_z
    planar_distance = max(float(np.hypot(delta_x, delta_y)), 1e-6)

    base_yaw = float(np.degrees(np.arctan2(delta_y, delta_x)))
    base_pitch = float(np.degrees(np.arctan2(delta_z, planar_distance)))

    rot_roll, rot_pitch, rot_yaw = rotation_offset
    return [base_pitch + rot_pitch, base_yaw + rot_yaw, rot_roll]


def set_free_view_near_pose(
    env: Any,
    focus_pose: List[float],
    offset: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
) -> None:
    """Place the native Unreal window near the UAV with an oblique side view."""
    cam_id = 0
    x, y, z = focus_pose[:3]
    yaw = focus_pose[4]

    offset_x, offset_y, offset_z = offset
    yaw_rad = np.radians(yaw)
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)
    cam_x = x + offset_x * cos_yaw - offset_y * sin_yaw
    cam_y = y + offset_x * sin_yaw + offset_y * cos_yaw
    cam_z = z + offset_z

    cam_pos = (cam_x, cam_y, cam_z)
    target_pos = (x, y, z)
    cam_rot = compute_look_at_rotation(cam_pos, target_pos, rotation)
    env.unwrapped.unrealcv.set_location(cam_id, list(cam_pos))
    env.unwrapped.unrealcv.set_cam_rotation(cam_id, cam_rot)


def get_follow_preview_cam_id(env: Any, policy_cam_id: int) -> int:
    """Pick a secondary camera id for third-person preview."""
    cam_config = env.unwrapped.unrealcv.get_camera_config()
    cam_ids = sorted(cam_config.keys())
    secondary_ids = [cam_id for cam_id in cam_ids if cam_id not in {0, policy_cam_id}]
    if secondary_ids:
        return secondary_ids[-1]
    return policy_cam_id


def get_third_person_preview_image(
    env: Any,
    cam_id: int,
    offset: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
) -> np.ndarray:
    """Capture a third-person follow shot of the drone using a secondary camera."""
    player = env.unwrapped.player_list[0]
    x, y, z = env.unwrapped.unrealcv.get_obj_location(player)
    roll, yaw, pitch = env.unwrapped.unrealcv.get_obj_rotation(player)

    offset_x, offset_y, offset_z = offset
    yaw_rad = np.radians(yaw)
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)
    cam_x = x + offset_x * cos_yaw - offset_y * sin_yaw
    cam_y = y + offset_x * sin_yaw + offset_y * cos_yaw
    cam_z = z + offset_z

    cam_pos = (cam_x, cam_y, cam_z)
    target_pos = (x, y, z)
    cam_rot = compute_look_at_rotation(cam_pos, target_pos, rotation)

    env.unwrapped.unrealcv.set_location(cam_id, list(cam_pos))
    env.unwrapped.unrealcv.set_cam_rotation(cam_id, cam_rot)
    return env.unwrapped.unrealcv.get_image(cam_id, 'lit')


def update_preview_window(
    image: np.ndarray,
    show_window: bool,
    preview_size: Tuple[int, int],
    window_name: str,
) -> None:
    """Render an image into a dedicated local preview window."""
    global UAV_PREVIEW_WINDOW_OPENED
    if not show_window:
        return
    preview = image
    if preview_size[0] > 0 and preview_size[1] > 0:
        preview = cv2.resize(preview, preview_size)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, preview)
    UAV_PREVIEW_WINDOW_OPENED = True
    cv2.waitKey(1)


def update_uav_preview_window(
    image: np.ndarray,
    show_window: bool,
    preview_size: Tuple[int, int],
) -> None:
    """Render the current UAV first-person image into a dedicated local window."""
    update_preview_window(image, show_window, preview_size, UAV_PREVIEW_WINDOW_NAME)


def get_player_focus_pose(env: Any) -> List[float]:
    """Get the current player pose as [x, y, z, roll, yaw]."""
    player = env.unwrapped.player_list[0]
    location = env.unwrapped.unrealcv.get_obj_location(player)
    rotation = env.unwrapped.unrealcv.get_obj_rotation(player)
    yaw = float(rotation[1]) if isinstance(rotation, (list, tuple)) and len(rotation) > 1 else 0.0
    return [float(location[0]), float(location[1]), float(location[2]), 0.0, yaw]


def get_preview_image(
    env: Any,
    policy_cam_id: int,
    uav_window_mode: str,
    third_person_cam_id: int,
    third_person_offset: Tuple[float, float, float],
    third_person_rotation: Tuple[float, float, float],
) -> np.ndarray:
    """Capture the image shown in the UAV preview window."""
    set_cam(env, policy_cam_id)
    image = env.unwrapped.unrealcv.get_image(policy_cam_id, 'lit')
    if uav_window_mode == 'third_person':
        return get_third_person_preview_image(env, third_person_cam_id, third_person_offset, third_person_rotation)
    return image


def run_preview_only(
    env: Any,
    viewport_mode: str,
    free_view_offset: Tuple[float, float, float],
    free_view_rotation: Tuple[float, float, float],
    show_uav_window: bool,
    uav_window_size: Tuple[int, int],
    uav_window_mode: str,
    policy_cam_id: int,
    third_person_cam_id: int,
    third_person_offset: Tuple[float, float, float],
    third_person_rotation: Tuple[float, float, float],
) -> None:
    """Keep the environment open for manual camera inspection without running tasks."""
    player_name = env.unwrapped.player_list[0]
    player_pose = get_player_focus_pose(env)
    if viewport_mode == 'free':
        set_free_view_near_pose(env, player_pose, free_view_offset, free_view_rotation)
    logger.info(f"Preview-only mode active. Player: {player_name}, pose: {player_pose[:3]}, yaw: {player_pose[4]:.2f}")
    logger.info("Preview-only mode keeps the environment open. Press Ctrl+C in the terminal to exit.")

    try:
        while True:
            if show_uav_window:
                preview_image = get_preview_image(
                    env,
                    policy_cam_id,
                    uav_window_mode,
                    third_person_cam_id,
                    third_person_offset,
                    third_person_rotation,
                )
                update_uav_preview_window(preview_image, True, uav_window_size)
            time.sleep(0.05)
    except KeyboardInterrupt:
        logger.info("Preview-only mode closed by user.")


def build_reference_action_sequence(reference_states: Optional[List[Any]]) -> List[List[float]]:
    """Convert reference local-frame states into action records compatible with control_loop."""
    if not isinstance(reference_states, list):
        return []

    action_sequence: List[List[float]] = []
    for state in reference_states[1:]:
        if not isinstance(state, (list, tuple)):
            continue
        if len(state) >= 5:
            x_rel = float(state[0])
            y_rel = float(state[1])
            z_rel = float(state[2])
            yaw_deg = float(state[4])
        elif len(state) >= 4:
            x_rel = float(state[0])
            y_rel = float(state[1])
            z_rel = float(state[2])
            yaw_deg = float(state[3])
        else:
            continue
        action_sequence.append([x_rel, y_rel, z_rel, float(np.radians(yaw_deg))])
    return action_sequence


def control_loop(
    initial_pos: List[float],
    env: Any,
    instruction: str,
    target_local: Optional[List[float]],
    max_steps: Optional[int],
    trajectory_log: List[Dict[str, Any]],
    server_url: str,
    show_uav_window: bool,
    uav_window_size: Tuple[int, int],
    policy_mode: str,
    reference_actions: Optional[List[List[float]]],
    uav_window_mode: str,
    policy_cam_id: int,
    third_person_cam_id: int,
    third_person_offset: Tuple[float, float, float],
    third_person_rotation: Tuple[float, float, float],
) -> None:
    """Main control loop: capture image/state, call inference, act in env, log trajectory."""
    initial_x, initial_y, initial_z = initial_pos[0:3]
    initial_yaw = initial_pos[4]
    env.unwrapped.unrealcv.set_obj_location(env.unwrapped.player_list[0], initial_pos[0:3])
    env.unwrapped.unrealcv.set_rotation(env.unwrapped.player_list[0], initial_pos[4] - 180)
    set_cam(env, policy_cam_id)
    time.sleep(SLEEP_AFTER_RESET_S)
    image = env.unwrapped.unrealcv.get_image(policy_cam_id, 'lit')
    if uav_window_mode == 'third_person':
        preview_image = get_third_person_preview_image(env, third_person_cam_id, third_person_offset, third_person_rotation)
    else:
        preview_image = image
    update_uav_preview_window(preview_image, show_uav_window, uav_window_size)
    
    current_pose: List[float] = [0, 0, 0, 0]
    logger.info('Start control loop!')
    last_pose: Optional[List[float]] = None
    small_count = 0
    step_count = 0
    reference_idx = 0

    def transform_to_global(x: float, y: float, initial_yaw: float) -> Tuple[float, float]:
        theta = np.radians(initial_yaw)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        global_x = x * cos_theta - y * sin_theta
        global_y = x * sin_theta + y * cos_theta
        return global_x, global_y

    def normalize_angle(angle: float) -> float:
        angle = angle % 360
        if angle > 180:
            angle -= 360
        return angle

    try:
        while True:
            set_cam(env, policy_cam_id)
            logger.debug(f"current_pose: {current_pose}")
            if policy_mode == 'reference':
                if not reference_actions or reference_idx >= len(reference_actions):
                    logger.info("Reference trajectory completed.")
                    break
                response = {'action': [reference_actions[reference_idx]]}
                reference_idx += 1
            else:
                t1 = time.time()
                response = send_prediction_request(
                    image=Image.fromarray(image),
                    proprio=np.array(current_pose),
                    instr=instruction,
                    server_url=server_url,
                    target_local=target_local,
                )
                t2 = time.time()
                # logger.info(f"Prediction time: {t2 - t1} seconds")
                if response is None:
                    logger.warning("No valid response, ending control")
                    break
            
            try:
                action_poses = response.get('action')
                if not isinstance(action_poses, list) or len(action_poses) == 0:
                    logger.warning("Response 'action' is empty or invalid, stopping.")
                    break
                for i, action_pose in enumerate(action_poses):
                    if not (isinstance(action_pose, (list, tuple)) and len(action_pose) >= 4):
                        logger.warning(f"Invalid action element at {i}: {action_pose}")
                        continue
                    relative_x, relative_y = float(action_pose[0]), float(action_pose[1])
                    relative_z = float(action_pose[2])
                    relative_yaw = float(np.degrees(action_pose[3]))
                    relative_yaw = (relative_yaw + 180) % 360 - 180
                    global_x, global_y = transform_to_global(relative_x, relative_y, initial_yaw)
                    absolute_yaw = normalize_angle(relative_yaw + initial_yaw)
                    absolute_pos = [
                        global_x + initial_x,
                        global_y + initial_y,
                        relative_z + initial_z,
                        absolute_yaw
                    ]
                    env.unwrapped.unrealcv.set_obj_location(env.unwrapped.player_list[0], absolute_pos[:3])
                    env.unwrapped.unrealcv.set_rotation(env.unwrapped.player_list[0], absolute_pos[3] - 180)
                    set_cam(env, policy_cam_id)
                    if i == len(action_poses) - 1:
                        current_pose = [relative_x, relative_y, relative_z, relative_yaw]
                        image = env.unwrapped.unrealcv.get_image(policy_cam_id, 'lit')
                        if uav_window_mode == 'third_person':
                            preview_image = get_third_person_preview_image(env, third_person_cam_id, third_person_offset, third_person_rotation)
                        else:
                            preview_image = image
                        update_uav_preview_window(preview_image, show_uav_window, uav_window_size)
                        try:
                            cv2.imwrite(DEBUG_IMAGE_PATH, image)
                        except Exception as e:
                            logger.debug(f"Failed to write debug image: {e}")
                    step_count += 1
                    trajectory_log.append({'state': [[relative_x, relative_y, relative_z], [0, relative_yaw, 0]]})
                    pose_now = [relative_x, relative_y, relative_z, relative_yaw]
                    if last_pose is not None:
                        diffs = [abs(a - b) for a, b in zip(pose_now, last_pose)]
                        if all(d < ACTION_SMALL_DELTA_POS for d in diffs[:3]) and diffs[3] < ACTION_SMALL_DELTA_YAW:
                            small_count += 1
                        else:
                            small_count = 0
                        if small_count >= ACTION_SMALL_STEPS:
                            logger.info(f"Detected x,y,z,yaw continuous {ACTION_SMALL_STEPS} steps change is very small, automatically end task.")
                            return
                    last_pose = pose_now
                    time.sleep(0.1)
                if max_steps is not None and step_count >= max_steps:
                    logger.info(f"Already inferred {max_steps} steps, automatically switch to next task.")
                    break
            except Exception as e:
                logger.error(f"Error executing action: {e}")
                break
            try:
                if response.get('done') is True:
                    logger.info("Server returned done=True. Ending control loop.")
                    return
            except Exception:
                pass
    except Exception as e:
        logger.error(f"Control loop error: {e}")


def set_cam(env: Any, cam_id: int) -> None:
    """Attach/sync the policy camera to the controlled drone."""
    player = env.unwrapped.player_list[0]
    agent_cfg = env.unwrapped.agents.get(player, {})
    rel_loc = agent_cfg.get('relative_location', [0, 0, 0])
    rel_rot = agent_cfg.get('relative_rotation', [0, 0, 0])
    env.unwrapped.unrealcv.set_cam(player, rel_loc, rel_rot)


def create_obj_if_needed(env: Any, obj_info: Optional[Dict[str, Any]]) -> None:
    """Create or place objects in the scene if needed."""
    if obj_info is None:
        return
    use_obj = obj_info.get('use_obj', None)
    obj_id = obj_info.get('obj_id', None)
    obj_pos = obj_info.get('obj_pos', None)
    obj_rot = obj_info.get('obj_rot', None)
    if use_obj == 1:
        env.unwrapped.unrealcv.set_appearance("BP_Character_21", obj_id)
        env.unwrapped.unrealcv.set_obj_location("BP_Character_21", obj_pos)
        env.unwrapped.unrealcv.set_obj_rotation("BP_Character_21", obj_rot)
        env.unwrapped.unrealcv.set_obj_location("BP_Character_22", [0, 0, -1000])
        env.unwrapped.unrealcv.set_obj_location("BP_Character_21", obj_pos)
    elif use_obj == 2:
        env.unwrapped.unrealcv.set_appearance("BP_Character_22", 2)
        env.unwrapped.unrealcv.set_obj_location("BP_Character_22", [obj_pos[0], obj_pos[1], 0])
        env.unwrapped.unrealcv.set_obj_rotation("BP_Character_22", obj_rot)
        env.unwrapped.unrealcv.set_phy("BP_Character_22", 0)
        env.unwrapped.unrealcv.set_obj_location("BP_Character_21", [0, 0, -1000])
        env.unwrapped.unrealcv.set_obj_location("BP_Character_22", [obj_pos[0], obj_pos[1], 0])

    if use_obj in [1, 2]:
        try:
            if use_obj == 1:
                logger.info(f"Spawned person target at: {env.unwrapped.unrealcv.get_obj_location('BP_Character_21')}")
            elif use_obj == 2:
                logger.info(f"Spawned vehicle target at: {env.unwrapped.unrealcv.get_obj_location('BP_Character_22')}")
        except Exception as e:
            logger.warning(f"Failed to read spawned target location: {e}")
        time.sleep(SLEEP_SHORT_S)


def reset_model(server_url: str) -> None:
    """Call server /reset to reset the model."""
    try:
        resp = requests.post(server_url.replace('/predict', '/reset'), timeout=10)
        logger.info(f"Model reset response: {resp.status_code}")
    except Exception as e:
        logger.error(f"Model reset failed: {e}")


def world_to_local(target: List[float], init_pos: List[float]) -> List[float]:
    """Convert world-frame target coordinates to the first-frame local frame."""
    x0, y0, z0 = init_pos[0:3]
    yaw0 = init_pos[4]
    dx = target[0] - x0
    dy = target[1] - y0
    dz = target[2] - z0
    theta = -np.radians(yaw0)
    x_rel = dx * np.cos(theta) - dy * np.sin(theta)
    y_rel = dx * np.sin(theta) + dy * np.cos(theta)
    z_rel = dz
    return [x_rel, y_rel, z_rel]


def resolve_setting_file_from_env_id(env_id: str) -> Optional[str]:
    """Map a Gym env_id to its JSON setting file under gym_unrealcv/envs/setting."""
    task_names = ["NavigationMulti", "Navigation", "Rendezvous", "Rescue", "Track"]
    suffix_pattern = r"-(Discrete|Continuous|Mixed)(ColorMask|MaskDepth|Rgbd|Color|Depth|Gray|CG|Mask|Pose)-v\d+$"
    suffix_match = re.search(suffix_pattern, env_id)
    if suffix_match is None:
        return None

    for task_name in task_names:
        prefix = f"Unreal{task_name}-"
        if not env_id.startswith(prefix):
            continue
        env_name = env_id[len(prefix):suffix_match.start()]
        if not env_name:
            return None
        return os.path.join(task_name, f"{env_name}.json")
    return None


def maybe_override_env_binary(env_id: str, env_bin_win: Optional[str]) -> None:
    """Override env_bin_win in the mapped setting JSON when requested."""
    if not env_bin_win:
        return

    setting_file = resolve_setting_file_from_env_id(env_id)
    if setting_file is None:
        raise ValueError(f"Cannot infer setting file from env_id: {env_id}")

    setting_path = misc.get_settingpath(setting_file)
    if not os.path.exists(setting_path):
        raise FileNotFoundError(f"Setting file not found: {setting_path}")

    env_bin_win = os.path.normpath(os.path.abspath(os.path.expanduser(env_bin_win)))
    if not os.path.exists(env_bin_win):
        raise FileNotFoundError(f"Environment binary not found: {env_bin_win}")

    with open(setting_path, "r", encoding="utf-8") as f:
        setting = json.load(f)
    setting["env_bin_win"] = env_bin_win
    with open(setting_path, "w", encoding="utf-8") as f:
        json.dump(setting, f, indent=4)
        f.write("\n")

    logger.info(f"Updated env_bin_win for {env_id} to: {env_bin_win}")


def validate_env_binary_exists(env_id: str) -> None:
    """Check whether the current env_bin_win configured for env_id exists."""
    setting_file = resolve_setting_file_from_env_id(env_id)
    if setting_file is None:
        return

    setting_path = misc.get_settingpath(setting_file)
    if not os.path.exists(setting_path):
        return

    with open(setting_path, "r", encoding="utf-8") as f:
        setting = json.load(f)

    env_bin_win = setting.get("env_bin_win")
    if sys.platform.startswith("win") and env_bin_win:
        env_bin_win = os.path.normpath(os.path.abspath(os.path.expanduser(env_bin_win)))
        if not os.path.exists(env_bin_win):
            raise FileNotFoundError(
                "Configured env_bin_win does not exist. "
                f"Current value: {env_bin_win}. "
                "Pass --env_bin_win with the correct Collection.exe path."
            )


def configure_player_viewport(
    env: Any,
    viewport_mode: str,
    viewport_offset: Tuple[float, float, float],
    viewport_rotation: Tuple[float, float, float],
) -> None:
    """Configure the local game viewport camera attached to the controlled drone."""
    player = env.unwrapped.player_list[0]
    if viewport_mode == "free":
        logger.info("Keeping the native Unreal window in free viewport mode.")
        return
    if viewport_mode == "third_person":
        env.unwrapped.unrealcv.set_cam(player, list(viewport_offset), list(viewport_rotation))
    else:
        env.unwrapped.unrealcv.set_cam(player, [0, 0, 0], [0, 0, 0])
    env.unwrapped.unrealcv.set_viewport(player)

if __name__ == '__main__':
    # ====== Default parameters ======
    DEFAULT_ENV_ID = 'UnrealTrack-DowntownWest-ContinuousColor-v0'
    DEFAULT_TIME_DILATION = 10
    DEFAULT_SEED = 0
    DEFAULT_JSON_FOLDER = r'./test_jsons'
    DEFAULT_SERVER_PORT = 5007
    DEFAULT_MAX_STEPS = 100
    DEFAULT_INSTRUCTION_TYPE = "instruction"
    # Fixed tick spacing (grid size)
    DEFAULT_TICK_X: Optional[float] = 100
    DEFAULT_TICK_Y: Optional[float] = 100
    DEFAULT_TICK_Z: Optional[float] = 100
    # Minimum display span
    DEFAULT_MIN_SPAN_X: Optional[float] = 400
    DEFAULT_MIN_SPAN_Y: Optional[float] = 400
    DEFAULT_MIN_SPAN_Z: Optional[float] = 400

    import argparse
    parser = argparse.ArgumentParser(description='Batch run UAV control with instruction-conditioned policy')
    parser.add_argument("-e", "--env_id", default=DEFAULT_ENV_ID, help='Environment ID to run')
    parser.add_argument("-t", '--time_dilation', default=DEFAULT_TIME_DILATION, type=int, help='Time dilation parameter to keep FPS in simulator')
    parser.add_argument("-s", '--seed', default=DEFAULT_SEED, type=int, help='Random seed')
    parser.add_argument("-f", '--json_folder', default=DEFAULT_JSON_FOLDER, help='Folder path containing batch task json files')
    parser.add_argument("-o", '--images_dir', default=None, help='Directory to save images and trajectory logs')
    parser.add_argument("-p", '--server_port', default=DEFAULT_SERVER_PORT, type=int, help='Inference server port')
    parser.add_argument("-m", '--max_steps', default=DEFAULT_MAX_STEPS, type=int, help='Maximum inference steps')
    parser.add_argument("-i", "--instruction_type", default=DEFAULT_INSTRUCTION_TYPE, choices=["instruction", "instruction_unified"], help='Choose which field to use: instruction or instruction_unified')
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], help='Logging level')
    parser.add_argument('--env_bin_win', default=None, help='Override env_bin_win in the mapped environment setting JSON')
    parser.add_argument('--window_width', type=int, default=256, help='Window/capture width used by UnrealCV')
    parser.add_argument('--window_height', type=int, default=256, help='Window/capture height used by UnrealCV')
    parser.add_argument('--viewport_mode', default='first_person', choices=['first_person', 'third_person', 'free'], help='Local viewport camera mode')
    parser.add_argument('--viewport_offset_x', type=float, default=-220.0, help='Third-person viewport X offset relative to the drone')
    parser.add_argument('--viewport_offset_y', type=float, default=0.0, help='Third-person viewport Y offset relative to the drone')
    parser.add_argument('--viewport_offset_z', type=float, default=90.0, help='Third-person viewport Z offset relative to the drone')
    parser.add_argument('--viewport_roll', type=float, default=0.0, help='Third-person viewport roll')
    parser.add_argument('--viewport_pitch', type=float, default=-12.0, help='Third-person viewport pitch')
    parser.add_argument('--viewport_yaw', type=float, default=0.0, help='Third-person viewport yaw')
    parser.add_argument('--free_view_offset_x', type=float, default=-220.0, help='Free-view startup camera X offset relative to the UAV')
    parser.add_argument('--free_view_offset_y', type=float, default=140.0, help='Free-view startup camera Y offset relative to the UAV')
    parser.add_argument('--free_view_offset_z', type=float, default=50.0, help='Free-view startup camera Z offset relative to the UAV')
    parser.add_argument('--free_view_roll', type=float, default=0.0, help='Free-view startup camera roll')
    parser.add_argument('--free_view_pitch', type=float, default=0.0, help='Free-view startup camera pitch offset after auto-look-at')
    parser.add_argument('--free_view_yaw', type=float, default=0.0, help='Free-view startup camera yaw offset')
    parser.add_argument('--show_uav_window', action='store_true', help='Show a separate OpenCV window for the UAV first-person view')
    parser.add_argument('--uav_window_width', type=int, default=960, help='Width of the UAV first-person preview window')
    parser.add_argument('--uav_window_height', type=int, default=540, help='Height of the UAV first-person preview window')
    parser.add_argument('--uav_window_mode', default='auto', choices=['auto', 'first_person', 'third_person'], help='Preview mode for the UAV OpenCV window')
    parser.add_argument('--uav_window_offset_x', type=float, default=-260.0, help='Third-person UAV preview X offset relative to the drone')
    parser.add_argument('--uav_window_offset_y', type=float, default=0.0, help='Third-person UAV preview Y offset relative to the drone')
    parser.add_argument('--uav_window_offset_z', type=float, default=120.0, help='Third-person UAV preview Z offset relative to the drone')
    parser.add_argument('--uav_window_roll', type=float, default=0.0, help='Third-person UAV preview roll')
    parser.add_argument('--uav_window_pitch', type=float, default=-12.0, help='Third-person UAV preview pitch')
    parser.add_argument('--uav_window_yaw', type=float, default=0.0, help='Third-person UAV preview yaw offset')
    parser.add_argument('--hold_windows', action='store_true', help='Keep the UAV preview window open until a key is pressed after all tasks finish')
    parser.add_argument('--force_run', action='store_true', help='Run tasks even if output trajectory/images already exist')
    parser.add_argument('--policy_mode', default='server', choices=['server', 'reference'], help='Use the model server or replay reference_path_preprocessed from the task JSON')
    parser.add_argument('--preview_only', action='store_true', help='Only launch the environment and preview cameras without running any task JSONs')
    # Fixed tick spacing (grid size)
    parser.add_argument('--tick_x', type=float, default=None, help='Tick step for X (forward) axis')
    parser.add_argument('--tick_y', type=float, default=None, help='Tick step for Y (right) axis')
    parser.add_argument('--tick_z', type=float, default=None, help='Tick step for Z (up) axis')
    # Minimum display span
    parser.add_argument('--min_span_x', type=float, default=None, help='Minimum span for X (forward) axis')
    parser.add_argument('--min_span_y', type=float, default=None, help='Minimum span for Y (right) axis')
    parser.add_argument('--min_span_z', type=float, default=None, help='Minimum span for Z (up) axis')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s')

    tick_x = args.tick_x if args.tick_x is not None else DEFAULT_TICK_X
    tick_y = args.tick_y if args.tick_y is not None else DEFAULT_TICK_Y
    tick_z = args.tick_z if args.tick_z is not None else DEFAULT_TICK_Z

    min_span_x = args.min_span_x if args.min_span_x is not None else DEFAULT_MIN_SPAN_X
    min_span_y = args.min_span_y if args.min_span_y is not None else DEFAULT_MIN_SPAN_Y
    min_span_z = args.min_span_z if args.min_span_z is not None else DEFAULT_MIN_SPAN_Z

    server_url = f"http://127.0.0.1:{args.server_port}/predict"
    images_dir = args.images_dir or f'./results/{args.env_id}/openvla'
    viewport_offset = (args.viewport_offset_x, args.viewport_offset_y, args.viewport_offset_z)
    viewport_rotation = (args.viewport_roll, args.viewport_pitch, args.viewport_yaw)
    free_view_offset = (args.free_view_offset_x, args.free_view_offset_y, args.free_view_offset_z)
    free_view_rotation = (args.free_view_roll, args.free_view_pitch, args.free_view_yaw)
    uav_window_size = (args.uav_window_width, args.uav_window_height)
    uav_window_mode = args.uav_window_mode
    if uav_window_mode == 'auto':
        uav_window_mode = 'first_person'
    uav_window_offset = (args.uav_window_offset_x, args.uav_window_offset_y, args.uav_window_offset_z)
    uav_window_rotation = (args.uav_window_roll, args.uav_window_pitch, args.uav_window_yaw)

    maybe_override_env_binary(args.env_id, args.env_bin_win)
    validate_env_binary_exists(args.env_id)

    env = gym.make(args.env_id)
    if int(args.time_dilation) > 0:
        env = time_dilation.TimeDilationWrapper(env, int(args.time_dilation))
    env.unwrapped.agents_category = ['drone']
    env = configUE.ConfigUEWrapper(env, resolution=(args.window_width, args.window_height))
    env = augmentation.RandomPopulationWrapper(env, 2, 2, random_target=False)
    env.seed(int(args.seed))
    env.reset()
    configure_player_viewport(env, args.viewport_mode, viewport_offset, viewport_rotation)
    policy_cam_id = get_policy_cam_id(env)
    third_person_cam_id = get_follow_preview_cam_id(env, policy_cam_id)
    logger.info(f"Using policy_cam_id={policy_cam_id}, preview_cam_id={third_person_cam_id}, viewport_mode={args.viewport_mode}, uav_window_mode={uav_window_mode}")
    env.unwrapped.unrealcv.set_phy(env.unwrapped.player_list[0], 0)
    logger.info(env.unwrapped.unrealcv.get_camera_config())
    print(env.unwrapped.unrealcv.get_camera_config())

    if args.preview_only:
        run_preview_only(
            env=env,
            viewport_mode=args.viewport_mode,
            free_view_offset=free_view_offset,
            free_view_rotation=free_view_rotation,
            show_uav_window=args.show_uav_window,
            uav_window_size=uav_window_size,
            uav_window_mode=uav_window_mode,
            policy_cam_id=policy_cam_id,
            third_person_cam_id=third_person_cam_id,
            third_person_offset=uav_window_offset,
            third_person_rotation=uav_window_rotation,
        )
        if args.show_uav_window and UAV_PREVIEW_WINDOW_OPENED:
            cv2.destroyAllWindows()
        env.close()
        sys.exit(0)

    json_folder = args.json_folder
    json_files = glob.glob(os.path.join(json_folder, '*.json'))
    logger.info(f"Detected {len(json_files)} json task files")
    os.makedirs(images_dir, exist_ok=True)

    # init object
    time.sleep(SLEEP_SHORT_S)
    env.unwrapped.unrealcv.new_obj("bp_character_C", "BP_Character_21", [0,0,0])
    env.unwrapped.unrealcv.set_appearance("BP_Character_21", 0)
    env.unwrapped.unrealcv.set_obj_rotation("BP_Character_21",  [0,0,0])
    time.sleep(SLEEP_SHORT_S)
    env.unwrapped.unrealcv.new_obj("BP_BaseCar_C", "BP_Character_22", [1000,0,0])
    env.unwrapped.unrealcv.set_appearance("BP_Character_22", 2)
    env.unwrapped.unrealcv.set_obj_rotation("BP_Character_22", [0,0,0])
    env.unwrapped.unrealcv.set_phy("BP_Character_22", 0)
    time.sleep(SLEEP_SHORT_S)


    for idx, json_file in enumerate(json_files):

        base_name = os.path.splitext(os.path.basename(json_file))[0]
        img2d_path = os.path.join(images_dir, base_name + '_2d.png')
        img3d_path = os.path.join(images_dir, base_name + '_3d.png')
        if (not args.force_run) and os.path.exists(img2d_path) and os.path.exists(img3d_path):
            logger.info(f"{base_name} images already exist, skipping.")
            continue
        logger.info(f"\n===== Start task {idx+1}/{len(json_files)} file: {json_file} =====")
        if args.policy_mode == 'server':
            reset_model(server_url)
        with open(json_file, 'r') as f:
            manual_data = json.load(f)
        if not isinstance(manual_data, dict):
            logger.warning(f"Unsupported json format, skipping: {json_file}")
            continue
        # Choose instruction field based on argument (compatible with legacy/new format)
        if args.instruction_type == 'instruction_unified':
            instruction = manual_data.get('instruction_unified', manual_data.get('instruction', ''))
        else:
            instruction = manual_data.get('instruction', manual_data.get('instruction_unified', ''))
        initial_pos = manual_data.get('initial_pos', None)
        target_pos = manual_data.get('target_pos', None)
        # Only create objects when both obj_id and use_obj are provided
        obj_info = None
        if 'obj_id' in manual_data and 'use_obj' in manual_data:
            if 'target_pos' in manual_data and isinstance(manual_data['target_pos'], list) and len(manual_data['target_pos']) == 6:
                obj_pos = manual_data['target_pos'][:3]
                obj_rot = manual_data['target_pos'][3:]
            else:
                obj_pos = manual_data.get('obj_pos', None)
                obj_rot = manual_data.get('obj_rot', [0, 0, 0])
            if obj_pos is not None:
                obj_info = {
                    'use_obj': manual_data['use_obj'],
                    'obj_id': manual_data['obj_id'],
                    'obj_pos': obj_pos,
                    'obj_rot': obj_rot
                }
        create_obj_if_needed(env, obj_info)
        set_cam(env, policy_cam_id)
        time.sleep(SLEEP_SHORT_S)
        logger.info(f"instruction: {instruction}")
        trajectory_log: List[Dict[str, Any]] = []
        player_name = env.unwrapped.player_list[0]
        try:
            logger.info(f"Control player: {player_name}")
            logger.info(f"Control player location before task: {env.unwrapped.unrealcv.get_obj_location(player_name)}")
            if obj_info is not None and obj_info.get('obj_pos') is not None:
                logger.info(f"Target object location: {obj_info['obj_pos']}")
        except Exception as e:
            logger.warning(f"Failed to print debug positions: {e}")
        reference_actions = build_reference_action_sequence(manual_data.get('reference_path_preprocessed'))
        if not initial_pos or len(initial_pos) < 5:
            logger.error("Invalid or missing 'initial_pos' in task json; skipping.")
            continue
        target_local = None
        if target_pos is not None:
            try:
                target_local = world_to_local(target_pos, initial_pos)
                logger.info(f"Target local position: {target_local}")
            except Exception as e:
                logger.warning(f"Failed to compute target_local: {e}")
        env.unwrapped.unrealcv.set_obj_location(player_name, initial_pos[0:3])
        env.unwrapped.unrealcv.set_rotation(player_name, initial_pos[4] - 180)
        if args.viewport_mode == 'free':
            set_free_view_near_pose(env, initial_pos, free_view_offset, free_view_rotation)
        control_loop(
            initial_pos,
            env=env,
            instruction=instruction,
            target_local=target_local,
            max_steps=args.max_steps,
            trajectory_log=trajectory_log,
            server_url=server_url,
            show_uav_window=args.show_uav_window,
            uav_window_size=uav_window_size,
            policy_mode=args.policy_mode,
            reference_actions=reference_actions,
            uav_window_mode=uav_window_mode,
            policy_cam_id=policy_cam_id,
            third_person_cam_id=third_person_cam_id,
            third_person_offset=uav_window_offset,
            third_person_rotation=uav_window_rotation,
        )
        traj_json_path = os.path.join(images_dir, base_name + '.json')
        with open(traj_json_path, 'w') as f:
            json.dump(trajectory_log, f, indent=2)
        # Transform target_pos to the coordinate system of the first frame
        try:
            draw_2d_trajectory_from_log(traj_json_path, os.path.join(images_dir, base_name + '_2d.png'), instruction, target_local, tick_x=tick_x, tick_y=tick_y, min_span_x=min_span_x, min_span_y=min_span_y)
            draw_3d_trajectory_from_log(traj_json_path, os.path.join(images_dir, base_name + '_3d.png'), instruction, target_local, tick_x=tick_x, tick_y=tick_y, tick_z=tick_z, min_span_x=min_span_x, min_span_y=min_span_y, min_span_z=min_span_z)
            logger.info(f"Trajectory and images saved: {base_name}")
        except Exception as e:
            logger.error(f"Plotting failed: {e}")
        logger.info(f"===== Task {idx+1} finished =====\n")
    if args.show_uav_window and args.hold_windows and UAV_PREVIEW_WINDOW_OPENED:
        logger.info("Press any key in the UAV preview window to close it.")
        cv2.waitKey(0)
    if args.show_uav_window and UAV_PREVIEW_WINDOW_OPENED:
        cv2.destroyAllWindows()
    env.close() 
