import mujoco

xml_path = "scene_base.xml"

model = mujoco.MjModel.from_xml_path(xml_path)
