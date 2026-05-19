import mujoco
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

xml_path = "assets/robots/stanford_tidybot/test01.py"

model = mujoco.MjModel.from_xml_path(xml_path)
logging.info("Model loaded successfully.")

data = mujoco.MjData(model)

