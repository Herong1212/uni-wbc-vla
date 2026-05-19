import mujoco
import mujoco.viewer
import logging
import mediapy as media
from termcolor import colored
import numpy as np
import time

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_model(xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    logging.info("Model loaded successfully.")

    return model, data


def print_model_attr(model):
    print("Model attributes:")

    # case1 查看模型的 geom 属性
    print(f"模型的 geom 数量：", model.ngeom)  # 11
    print(f"-- 模型的几何体名称为: ", [model.geom(i).name for i in range(model.ngeom)])
    # [
    #     "floor",                          # 地面
    #     "bumper_visual",                  # 保险杠视觉模型
    #     "bottom_plate_visual",            # 底板视觉模型
    #     "body_visual",                    # 主体视觉模型
    #     "top_plate_visual",               # 顶板视觉模型
    #     "arm_plate_visual",               # 机械臂基板视觉模型
    #     "bumper_collision",               # 保险杠碰撞模型
    #     "bottom_plate_collision",         # 底板碰撞模型
    #     "body_collision",                 # 主体碰撞模型
    #     "top_plate_collision",            # 顶板碰撞模型
    #     "arm_plate_collision",            # 机械臂基板碰撞模型
    # ]

    for i in range(model.ngeom):
        if model.geom(i) == model.geom("floor"):
            continue
        model.geom(i).rgba[:3] = np.random.rand(3)  # 随机修改 geom 的 RGB 值
        print(f"{model.geom(i).name} 的颜色为：{model.geom(i).rgba}")

    # case2 查看模型的 body 属性
    print(f"模型的 body 数量：", model.nbody)  # 2
    print(f"-- 模型的身体名称为: ", [model.body(i).name for i in range(model.nbody)])
    # ["world", "base_link"]

    # case3 查看其它属性
    print(f"模型的 nq 数量：", model.nq)  # 3 - 广义坐标数量（位置自由度）
    print(f"模型的 nv 数量：", model.nv)  # 3 - 广义速度数量（速度自由度）
    print(f"模型的 light 数量：", model.nlight)  # 1 - 光源数量
    print(f"模型的 camera 数量：", model.ncam)  # 1 - 相机数量

    # case4 查看关节属性
    # print(f"模型的 joint 数量：", model.njoint)


def main():
    # ? logging.info 和 print 的区别？
    logging.info(
        colored(
            "###### Starting the simulation! ######",
            "yellow",
            attrs=["bold"],
        )
    )

    xml_path = "/home/robot/uni-wbc-vla/assets/arenas/pickandplace/scene.xml"
    # xml_path = "/home/robot/uni-wbc-vla/assets/robots/stanford_tidybot/scene.xml"
    model, data = load_model(xml_path)

    print_model_attr(model)

    # 函数原型：MJAPI int mj_name2id(const mjModel *m, int type, const char *name);
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "")

    # case1 重置到 "home" 关键帧（索引 0）
    # mujoco.mj_resetData(model, data)  # Reset state and time.
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # case2 或者重置到 "retract" 关键帧（索引 1）
    # mujoco.mj_resetDataKeyframe(model, data, 1)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.time()
            step_start = time.time()

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(model, data)

            # Example modification of a viewer option: toggle contact points every two seconds.
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(
                    data.time % 2
                )

            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
