import mujoco
import logging
import mediapy as media

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_model(xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    return model, data


def main():
    logging.info("Starting the simulation.")

    xml_path = "assets/robots/stanford_tidybot/scene_base.xml"
    model = load_model(xml_path)

    model, data = mujoco.MjModel.from_xml_path(xml_path)

    logging.info("Model loaded successfully.")

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

    data = mujoco.MjData(model)


# with mujoco.Renderer(model) as renderer:
# media.show_image(renderer.render())

if __name__ == "__main__":
    main()
