import numpy as np

from .mujoco_parser import MuJoCoParserClass
from .utils import sample_xyzs, rotation_matrix, add_title_to_img
from .ik import solve_ik
from .transforms import rpy2r, r2rpy
import copy
import glfw


class TidybotEnv:
    """
    Mobile Manipulation Environment for Stanford TidyBot.

    Robot: Mobile base (3-DOF: x, y, theta) + Kinova Gen3 arm (7-DOF) + Robotiq 2F-85 gripper.

    Action space (depends on action_type):
        - eef_pose (default): [dx, dy, dz, droll, dpitch, dyaw, gripper] (7-dim)
          Arm end-effector delta pose via IK. Base joints remain fixed.
        - delta_joint_angle: 11-dim [j1..j10, gripper] — joint angle deltas for all 10 joints
        - joint_angle: 11-dim [j1..j10, gripper] — absolute joint angles for all 10 joints
          Gripper: 0=open, 1=closed (mapped to fingers_actuator ctrl [0, 255])

    State space (depends on state_type):
        - joint_angle: 11-dim [j1..j10, gripper]
        - ee_pose: 6-dim [px, py, pz, roll, pitch, yaw]
        - delta_q: 11-dim [dj1..dj10, gripper]
    """

    def __init__(
        self, xml_path, action_type="eef_pose", state_type="joint_angle", seed=None
    ):
        self.env = MuJoCoParserClass(name="Tidybot", rel_xml_path=xml_path)
        self.action_type = action_type
        self.state_type = state_type

        # Joint groups
        self.base_joint_names = ["joint_x", "joint_y", "joint_th"]
        self.arm_joint_names = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "joint_7",
        ]
        self.all_joint_names = self.base_joint_names + self.arm_joint_names  # 10 joints

        # Robotiq 2F-85 gripper: tendon-driven general actuator (ctrl range [0, 255])
        # right_driver_joint qpos: 0 = open, 0.8 = closed
        self.gripper_joint_name = "right_driver_joint"

        # IK target: last arm body (before gripper)
        self.tcp_body_name = "bracelet_link"

        self.init_viewer()
        self.reset(seed)

    def init_viewer(self):
        self.env.reset()
        self.env.init_viewer(
            distance=2.0,
            elevation=-30,
            transparent=False,
            black_sky=True,
            use_rgb_overlay=False,
            loc_rgb_overlay="top right",
        )

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Arm home configuration (Kinova Gen3 ready pose from model keyframe)
        q_arm_init = np.array(
            [
                0.0,
                0.26179939,
                3.14159265,
                -2.26892803,
                0.0,
                0.95993109,
                1.57079633,
            ]
        )
        q_base = np.zeros(3)  # base at origin

        # IK with arm-only joints to reach initial EE pose
        q_arm, ik_err_stack, ik_info = solve_ik(
            env=self.env,
            joint_names_for_ik=self.arm_joint_names,
            body_name_trgt=self.tcp_body_name,
            q_init=q_arm_init,
            p_trgt=np.array([0.3, 0.0, 1.0]),
            R_trgt=rpy2r(np.deg2rad([90, -0.0, 90])),
        )
        q_zero = np.concatenate([q_base, q_arm])
        self.env.forward(
            q=q_zero, joint_names=self.all_joint_names, increase_tick=False
        )

        # Set object positions (objects with "body_obj_" prefix)
        obj_names = self.env.get_body_names(prefix="body_obj_")
        n_obj = len(obj_names)
        if n_obj > 0:
            obj_xyzs = sample_xyzs(
                n_obj,
                x_range=[+0.24, +0.4],
                y_range=[-0.2, +0.2],
                z_range=[0.82, 0.82],
                min_dist=0.2,
                xy_margin=0.0,
            )
            for obj_idx in range(n_obj):
                self.env.set_p_base_body(
                    body_name=obj_names[obj_idx], p=obj_xyzs[obj_idx, :]
                )
                self.env.set_R_base_body(body_name=obj_names[obj_idx], R=np.eye(3, 3))
        self.env.forward(increase_tick=False)

        # Store initial state
        self.last_q = copy.deepcopy(q_zero)
        # Ctrl vector: 3 base + 7 arm + 1 gripper = 11
        self.q = np.concatenate([q_zero, [0.0]])  # gripper open (ctrl=0)
        self.p0, self.R0 = self.env.get_pR_body(body_name=self.tcp_body_name)

        # Record initial object poses
        obj_init_poses = []
        for name in obj_names:
            obj_init_poses.append(self.env.get_p_body(name))
        self.obj_init_pose = (
            np.concatenate(obj_init_poses, dtype=np.float32)
            if obj_init_poses
            else np.array([], dtype=np.float32)
        )

        for _ in range(100):
            self.step_env()
        print("DONE INITIALIZATION")
        self.gripper_state = False
        self.past_chars = []

    def step(self, action):
        """
        Take a step in the environment.

        args:
            action: np.array
                - eef_pose: shape (7,) = [dx, dy, dz, droll, dpitch, dyaw, gripper]
                - delta_joint_angle: shape (11,) = [j1..j10, gripper]
                - joint_angle: shape (11,) = [j1..j10, gripper]
        returns:
            state: np.array depending on state_type
        """
        if self.action_type == "eef_pose":
            # Arm EE delta pose control via IK (base fixed)
            q_base = self.env.get_qpos_joints(joint_names=self.base_joint_names)
            q_arm = self.env.get_qpos_joints(joint_names=self.arm_joint_names)

            self.p0 += action[:3]
            self.R0 = self.R0.dot(rpy2r(action[3:6]))

            q_arm, ik_err_stack, ik_info = solve_ik(
                env=self.env,
                joint_names_for_ik=self.arm_joint_names,
                body_name_trgt=self.tcp_body_name,
                q_init=q_arm,
                p_trgt=self.p0,
                R_trgt=self.R0,
                max_ik_tick=50,
                ik_stepsize=1.0,
                ik_eps=1e-2,
                ik_th=np.radians(5.0),
                render=False,
                verbose_warning=False,
            )
            q = np.concatenate([q_base, q_arm])
            gripper_cmd = float(action[6]) * 255.0

        elif self.action_type == "delta_joint_angle":
            q = action[:10] + self.last_q
            gripper_cmd = float(action[10]) * 255.0

        elif self.action_type == "joint_angle":
            q = action[:10]
            gripper_cmd = float(action[10]) * 255.0

        else:
            raise ValueError(f"action_type '{self.action_type}' not recognized")

        self.compute_q = q
        self.q = np.concatenate([q, [gripper_cmd]])

        if self.state_type == "joint_angle":
            return self.get_joint_state()
        elif self.state_type == "ee_pose":
            return self.get_ee_pose()
        elif self.state_type == "delta_q":
            return self.get_delta_q()
        else:
            raise ValueError(f"state_type '{self.state_type}' not recognized")

    def step_env(self):
        self.env.step(self.q)

    def grab_image(self):
        self.rgb_agent = self.env.get_fixed_cam_rgb(cam_name="agentview")
        self.rgb_ego = self.env.get_fixed_cam_rgb(cam_name="egocentric")
        self.rgb_side = self.env.get_fixed_cam_rgb(cam_name="sideview")
        return self.rgb_agent, self.rgb_ego

    def render(self, teleop=False):
        self.env.plot_time()
        p_current, R_current = self.env.get_pR_body(body_name=self.tcp_body_name)
        R_current = R_current @ np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        self.env.plot_sphere(p=p_current, r=0.02, rgba=[0.95, 0.05, 0.05, 0.5])
        self.env.plot_capsule(
            p=p_current, R=R_current, r=0.01, h=0.2, rgba=[0.05, 0.95, 0.05, 0.5]
        )
        rgb_egocentric_view = add_title_to_img(
            self.rgb_ego, text="Egocentric View", shape=(640, 480)
        )
        rgb_agent_view = add_title_to_img(
            self.rgb_agent, text="Agent View", shape=(640, 480)
        )
        self.env.viewer_rgb_overlay(rgb_agent_view, loc="top right")
        self.env.viewer_rgb_overlay(rgb_egocentric_view, loc="bottom right")
        if teleop:
            rgb_side_view = add_title_to_img(
                self.rgb_side, text="Side View", shape=(640, 480)
            )
            self.env.viewer_rgb_overlay(rgb_side_view, loc="top left")
            self.env.viewer_text_overlay(
                text1="Key Pressed", text2="%s" % (self.env.get_key_pressed_list())
            )
            self.env.viewer_text_overlay(
                text1="Key Repeated",
                text2="%s" % (self.env.get_key_repeated_list()),
            )
        self.env.render()

    def get_joint_state(self):
        qpos = self.env.get_qpos_joints(joint_names=self.all_joint_names)
        gripper = self.env.get_qpos_joint(self.gripper_joint_name)
        gripper_cmd = 1.0 if gripper[0] > 0.4 else 0.0
        return np.concatenate([qpos, [gripper_cmd]], dtype=np.float32)

    def get_delta_q(self):
        delta = self.compute_q - self.last_q
        self.last_q = copy.deepcopy(self.compute_q)
        gripper = self.env.get_qpos_joint(self.gripper_joint_name)
        gripper_cmd = 1.0 if gripper[0] > 0.4 else 0.0
        return np.concatenate([delta, [gripper_cmd]], dtype=np.float32)

    def get_ee_pose(self):
        p, R = self.env.get_pR_body(body_name=self.tcp_body_name)
        rpy = r2rpy(R)
        return np.concatenate([p, rpy], dtype=np.float32)

    def teleop_robot(self):
        """
        Teleoperate the robot (base + arm + gripper) using keyboard.
        Returns: (action, done) for eef_pose control mode.

        Base control (local frame, follows robot facing direction):
            I/K: forward/backward
            J/L: strafe left/right
            U/O: rotate CCW/CW

        Arm end-effector control:
            W/S: forward/backward   (x)
            A/D: left/right         (y)
            R/F: up/down            (z)
            Q/E: yaw rotation       (z-axis)
            UP/DOWN: pitch rotation (x-axis)
            LEFT/RIGHT: roll rotation (y-axis)

        Other:
            SPACE: toggle gripper
            Z: reset episode
        """
        # --- Base control (applied directly to simulation) ---
        fb, strafe, rot = 0.0, 0.0, 0.0

        if self.env.is_key_pressed_repeat(key=glfw.KEY_I):
            fb += 0.01
        if self.env.is_key_pressed_repeat(key=glfw.KEY_K):
            fb += -0.01
        if self.env.is_key_pressed_repeat(key=glfw.KEY_J):
            strafe += 0.01
        if self.env.is_key_pressed_repeat(key=glfw.KEY_L):
            strafe += -0.01
        if self.env.is_key_pressed_repeat(key=glfw.KEY_U):
            rot += 0.02
        if self.env.is_key_pressed_repeat(key=glfw.KEY_O):
            rot += -0.02

        if abs(fb) > 0 or abs(strafe) > 0 or abs(rot) > 0:
            q_base = self.env.get_qpos_joints(joint_names=self.base_joint_names)
            th = q_base[2]
            c, s = np.cos(th), np.sin(th)
            # Transform local frame (forward, left) to world frame
            q_base[0] += fb * c - strafe * s
            q_base[1] += fb * s + strafe * c
            q_base[2] += rot
            self.env.forward(
                q=q_base, joint_names=self.base_joint_names, increase_tick=False
            )
            # Update EE pose reference after base movement
            self.p0, self.R0 = self.env.get_pR_body(body_name=self.tcp_body_name)

        # --- Arm end-effector control ---
        dpos = np.zeros(3)
        drot = np.eye(3)

        # XY plane
        if self.env.is_key_pressed_repeat(key=glfw.KEY_S):
            dpos += np.array([0.007, 0.0, 0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_W):
            dpos += np.array([-0.007, 0.0, 0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_A):
            dpos += np.array([0.0, -0.007, 0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_D):
            dpos += np.array([0.0, 0.007, 0.0])

        # Z axis
        if self.env.is_key_pressed_repeat(key=glfw.KEY_R):
            dpos += np.array([0.0, 0.0, 0.007])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_F):
            dpos += np.array([0.0, 0.0, -0.007])

        # Rotation
        if self.env.is_key_pressed_repeat(key=glfw.KEY_LEFT):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_RIGHT):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_DOWN):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_UP):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_Q):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_E):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]

        if self.env.is_key_pressed_once(key=glfw.KEY_Z):
            return np.zeros(7, dtype=np.float32), True
        if self.env.is_key_pressed_once(key=glfw.KEY_SPACE):
            self.gripper_state = not self.gripper_state

        drot = r2rpy(drot)

        action = np.concatenate(
            [dpos, drot, np.array([self.gripper_state], dtype=np.float32)],
            dtype=np.float32,
        )
        return action, False

    def check_success(self):
        """
        Override in subclass with task-specific conditions.
        Default: returns False.
        """
        return False

    def get_obj_pose(self):
        obj_names = self.env.get_body_names(prefix="body_obj_")
        return [self.env.get_p_body(name) for name in obj_names]

    def set_obj_pose(self, poses):
        obj_names = self.env.get_body_names(prefix="body_obj_")
        for name, p in zip(obj_names, poses):
            self.env.set_p_base_body(body_name=name, p=p)
            self.env.set_R_base_body(body_name=name, R=np.eye(3, 3))
        self.step_env()
