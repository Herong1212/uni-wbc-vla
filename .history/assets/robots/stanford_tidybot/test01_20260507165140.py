import mujoco
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

xml_path = "assets/robots/stanford_tidybot/scene_base.xml"

model = mujoco.MjModel.from_xml_path(xml_path)
logging.info("Model loaded successfully.")

data = mujoco.MjData(model)

with mujoco.renderer as renderer:
    mujoco.mj_render(model, data, renderer)
