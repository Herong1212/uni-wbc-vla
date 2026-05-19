import mujoco
import logging
import mediapy as media

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

xml_path = "assets/robots/stanford_tidybot/scene_base.xml"

model = mujoco.MjModel.from_xml_path(xml_path)
logging.info("Model loaded successfully.")

data = mujoco.MjData(model)

with mujoco.renderer(model) as renderer:
    media.show_image(renderer.render())
