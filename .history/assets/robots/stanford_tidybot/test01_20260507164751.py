import mujoco
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

xml_path = "scene_base.xml"

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

logging.info()
