import mujoco
import logging
import mediapy as media

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

xml_path = "assets/robots/stanford_tidybot/scene_base.xml"

model = mujoco.MjModel.from_xml_path(xml_path)
logging.info("Model loaded successfully.")

# 查看 model 的属性
print(model.ngeom)  # 11
print([model.geom(i).name for i in range(model.ngeom)])

data = mujoco.MjData(model)


# with mujoco.Renderer(model) as renderer:
    # media.show_image(renderer.render())
