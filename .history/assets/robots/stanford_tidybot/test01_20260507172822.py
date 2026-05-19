import mujoco
import logging
import mediapy as media
from termcolor import colored
import numpy as np

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

    # 查看 model 的属性
    print(f"模型的 geom 数量：", model.ngeom)  # 11
    print(f"model geom: ", [model.geom(i).name for i in range(model.ngeom)])
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
    model.geom("bumper_visual").rgba[:3] = np.random.rand(3)  # 随机修改防撞条的 RGB 值
    print(model.geom("bumper_visual").rgba)  # 查看修改后的防撞条的 RGB 值

    print(f"模型的 body 数量：", model.nbody)  # 2
    print([model.body(i).name for i in range(model.nbody)])
    # ["world", "base_link"]

    print(f"模型的 nq 数量：", model.nq)  # 3 - 广义坐标数量（位置自由度）
    print(f"模型的 nv 数量：", model.nv)  # 3 - 广义速度数量（速度自由度）
    print(f"模型的 light 数量：", model.nlight)  # 1 - 光源数量
    print(f"模型的 camera 数量：", model.ncam)  # 1 - 相机数量


def render_model(model, data):
    with mujoco.Renderer(model) as renderer:
        # 执行前向动力学计算，更新所有派生量
        mujoco.mj_forward(model, data)
        # 更新渲染场景
        renderer.update_scene(data)
        # 渲染并显示
        media.show_image(renderer.render())


def main():
    # ? logging.info 和 print 的区别？
    logging.info(
        colored(
            "###### Starting the simulation! ######",
            "yellow",
            attrs=["bold"],
        )
    )

    xml_path = "assets/robots/stanford_tidybot/scene_base.xml"
    model, data = load_model(xml_path)

    print_model_attr(model)

    render_model(model, data)


if __name__ == "__main__":
    main()
