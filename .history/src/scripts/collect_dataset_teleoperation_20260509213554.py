# *
# NOTE
import sys
import random
import numpy as np
import os
from PIL import Image
from mujoco_env.y_env import SimpleEnv  # 自定义的 MuJoCo 环境封装类
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

#   这个脚本实现的是通过键盘遥控机器人收集演示数据的功能。具体来说：
#       1、在 MuJoCo 仿真环境中，用户通过键盘控制机械臂完成"将马克杯放在盘子上"的任务
#       2、实时记录机器人的视觉、状态和动作数据
#       3、将数据保存为 LeRobot 格式的数据集，用于后续训练模仿学习模型

SEED = 42  # 随机种子，设为固定值确保每次物体位置相同（便于复现）
REPO_NAME = "omy_pnp"  # 数据集仓库名称（omy表示机器人类型，pnp 表示pick-and-place任务）
NUM_DEMO = 3  # 要收集的演示数量（3个完整任务回合）
ROOT = "./outputs/datasets"  # 数据集保存的根目录路径

TASK_NAME = "Put mug cup on the plate"
xml_path = "./asset/example_scene_y.xml"

# * 定义环境
PnPEnv = SimpleEnv(xml_path, seed=SEED, state_type="joint_angle")

create_new = True
if os.path.exists(ROOT):
    print(f"Directory {ROOT} already exists.")
    ans = input("Do you want to delete it? (y/n) ")
    if ans == "y":
        import shutil

        shutil.rmtree(ROOT)
    else:
        create_new = False

if create_new:
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        root=ROOT,
        robot_type="omy",
        fps=20,  # 20 frames per second
        features={
            "observation.image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["state"],  # x, y, z, roll, pitch, yaw
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["action"],  # 6 joint angles and 1 gripper
            },
            "obj_init": {
                "dtype": "float32",
                "shape": (6,),
                "names": [
                    "obj_init"
                ],  # just the initial position of the object. Not used in training.
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
else:
    print("Load from previous dataset")
    dataset = LeRobotDataset(REPO_NAME, root=ROOT)

action = np.zeros(7)
episode_id = 0
record_flag = False  # 当机器人开始移动时开始记录

while PnPEnv.env.is_viewer_alive() and episode_id < NUM_DEMO:
    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        # 检查回合是否完成
        done = PnPEnv.check_success()
        if done:
            # 保存回合数据并重置环境
            dataset.save_episode()
            PnPEnv.reset(seed=SEED)
            episode_id += 1

        # 遥控机器人并获取带夹爪的末端执行器位姿变化
        action, reset = PnPEnv.teleop_robot()
        if not record_flag and sum(action) != 0:
            record_flag = True
            print("Start recording")
        if reset:
            # 重置环境并清除回合缓冲区，可以通过按'z'键完成
            PnPEnv.reset(seed=SEED)
            # PnPEnv.reset()
            dataset.clear_episode_buffer()
            record_flag = False

        # 步进环境
        # 获取末端执行器位姿和图像
        ee_pose = PnPEnv.get_ee_pose()
        agent_image, wrist_image = PnPEnv.grab_image()
        # 调整大小为 256x256
        agent_image = Image.fromarray(agent_image)
        wrist_image = Image.fromarray(wrist_image)
        agent_image = agent_image.resize((256, 256))
        wrist_image = wrist_image.resize((256, 256))
        agent_image = np.array(agent_image)
        wrist_image = np.array(wrist_image)
        joint_q = PnPEnv.step(action)
        if record_flag:
            # 将帧添加到数据集
            dataset.add_frame(
                {
                    "observation.image": agent_image,
                    "observation.wrist_image": wrist_image,
                    "observation.state": ee_pose,
                    "action": joint_q,
                    "obj_init": PnPEnv.obj_init_pose,
                    # "task": TASK_NAME,
                },
                task=TASK_NAME,
            )
        PnPEnv.render(teleop=True)

PnPEnv.env.close_viewer()

# 清理图像文件夹
import shutil

shutil.rmtree(dataset.root / "images")
