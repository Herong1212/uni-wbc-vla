import mujoco
import logging
import mediapy as media
from termcolor import colored

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
    print(model.ngeom)  # 11
    print([model.geom(i).name for i in range(model.ngeom)])
    # [
    #     "floor",
    #     "bumper_visual",
    #     "bottom_plate_visual",
    #     "body_visual",
    #     "top_plate_visual",
    #     "arm_plate_visual",
    #     "bumper_collision",
    #     "bottom_plate_collision",
    #     "body_collision",
    #     "top_plate_collision",
    #     "arm_plate_collision",
    # ]


def main():
    logging.info(
        colored(
            "###### Starting the simulation! ######",
            "yellow",
            attrs=["bold"],
        )
    )

    xml_path = "assets/robots/stanford_tidybot/scene_base.xml"
    model, data = load_model(xml_path)


if __name__ == "__main__":
    main()
