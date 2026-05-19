import mujoco
import mujoco.viewer
import logging

import time

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# 方法一：通过 body 名称设置位置
# robot_body_id = model.body("base_link").id
# logging.info(f"Robot body ID: {robot_body_id}")


def main():
    logging.info("Starting the simulation.")

    # 加载场景
    model = mujoco.MjModel.from_xml_path("turtorial.xml")
    # ps world 是默认的全局坐标系，所有其他物体都是相对于 world 定义的，所以它的 ID 是 0。
    print(model.body("world").id)
    print(model.geom("red_box"))
    print(model.geom("red_box").rgba)
    # print(model.body("base_link").id)  # 1，那 0 是谁的？
    data = mujoco.MjData(model)

    logging.info("Model loaded successfully.")

    # NOTE
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Close the viewer automatically after 30 wall-seconds.
        start = time.time()  # 记录开始时间
        while viewer.is_running() and time.time() - start < 30:
            step_start = time.time()

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(model, data)  # *

            # Example modification of a viewer option: toggle contact points every two seconds.
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(
                    data.time % 2
                )

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = model.opt.timestep - (time.time() - step_start)

            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
