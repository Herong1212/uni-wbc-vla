import mujoco
import mujoco.viewer
import logging
import mediapy as media
from termcolor import colored
import numpy as np
import time
import argparse

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_model(xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    logging.info("Model loaded successfully.")

    return model, data


def print_info(model):
    print("---------------------------------------------------------------------------")
    print("Model attributes:")

    # case 查看基本属性
    print("")
    parsed_strings = [s for s in model.names.split(b"\x00") if s]
    # print(parsed_strings)
    parsed_strings = [s.decode("utf-8") for s in parsed_strings]
    # print(parsed_strings)
    print(f"# name: ", parsed_strings[0])
    print(f"# dt: {model.opt.timestep}")
    print(f"# HZ: {int(1 / model.opt.timestep)}")
    print(f"# n_qpos: ", model.nq)  # 广义坐标数量（位置自由度）
    print(f"# n_qvel: ", model.nv)  # 广义速度数量（速度自由度）
    print(f"# n_qacc: ", model.nv)  # 加速度计数量
    print(f"# n_qctrl: ", model.nu)  # 执行器数量
    print(f"# n_light: ", model.nlight)  # 光源数量
    print(f"# n_camera: ", model.ncam)  # 相机数量

    # 查看 stl 文件尺寸
    import trimesh

    # 加载STL文件
    mesh = trimesh.load(
        "/home/robot/Wiki-GRx-MJCF/models/TCB/meshes/composite_robot/torso_lift_link.STL"
    )

    # 获取包围盒尺寸
    dimensions = mesh.extents  # XYZ方向的全尺寸
    print(f"Torso lift dimensions: {dimensions}")

    integrator = model.opt.integrator
    if integrator == mujoco.mjtIntegrator.mjINT_EULER:
        integrator_name = "EULER"
    elif integrator == mujoco.mjtIntegrator.mjINT_RK4:
        integrator_name = "RK4"
    elif integrator == mujoco.mjtIntegrator.mjINT_IMPLICIT:
        integrator_name = "IMPLICIT"
    elif integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
        integrator_name = "IMPLICITFAST"
    else:
        integrator_name = "UNKNOWN"
    print(f"# integrator: ", integrator_name)  # 积分器类型

    # case 查看 body 属性
    print("")
    print(f"# n_body: ", model.nbody)  # 2
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_idx)
        for body_idx in range(model.nbody)
    ]
    for body_idx, body_name in enumerate(body_names):
        print(f"body_{body_idx} name: {body_name}")

    # case 查看 geom 属性
    print("")
    print(f"# n_geom: ", model.ngeom)
    print(f"geom_names: ", [model.geom(i).name for i in range(model.ngeom)])

    # 随机修改 geom 的 RGB 值
    for i in range(model.ngeom):
        model.geom(i).rgba[:3] = np.random.rand(3)
    print(f"{model.geom(i).name} 的颜色为：{model.geom(i).rgba}")

    # case 查看 joint 属性
    print("")
    print(f"# n_joint: ", model.njnt)
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_idx)
        for joint_idx in range(model.njnt)
    ]
    for joint_idx, joint_name in enumerate(joint_names):
        print(f"joint_{joint_idx} name: {joint_name}")

    # case 查看 dof 属性
    print("")
    n_dof = model.nv  # degree of freedom (=number of columns of Jacobian)
    print("# n_dof:[%d] (=number of rows of Jacobian)" % (n_dof))
    dof_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_DOF, i) for i in range(model.nv)
    ]
    for dof_idx, dof_name in enumerate(dof_names):
        joint_name = joint_names[model.dof_jntid[dof_idx]]
        body_name = body_names[model.dof_bodyid[dof_idx]]
        print(
            f"dof_{dof_idx} {dof_name} attached to joint: {joint_name}  body: {body_name}"
        )

    # case 查看 actuators 属性
    print("")
    n_ctrl = model.nu  # number of actuators (or controls)
    print(f"# n_ctrl: {n_ctrl}")
    ctrl_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, ctrl_idx)
        for ctrl_idx in range(n_ctrl)
    ]
    ctrl_ranges = model.actuator_ctrlrange  # control range
    ctrl_mins = ctrl_ranges[:, 0]
    ctrl_maxs = ctrl_ranges[:, 1]
    ctrl_gears = model.actuator_gear[:, 0]  # gears

    for ctrl_idx, ctrl_name in enumerate(ctrl_names):
        print(
            f"ctrl_{ctrl_idx} {ctrl_name} range: {ctrl_mins[ctrl_idx]} ~ {ctrl_maxs[ctrl_idx]} gear: {ctrl_gears[ctrl_idx]}"
        )

    # case 查看 camera 属性
    print("")
    n_cam = model.ncam
    cam_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_idx)
        for cam_idx in range(n_cam)
    ]
    cams = []
    cam_fovs = []
    cam_viewports = []
    for cam_idx in range(n_cam):
        cam_name = cam_names[cam_idx]
        cam = mujoco.MjvCamera()
        cam.fixedcamid = model.cam(cam_name).id
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam_fov = model.cam_fovy[cam_idx]
        viewport = mujoco.MjrRect(0, 0, 800, 600)
        # Append
        cams.append(cam)
        cam_fovs.append(cam_fov)
        cam_viewports.append(viewport)
    print(f"# n_cam: {n_cam}")
    for cam_idx, cam_name in enumerate(cam_names):
        print(f"cam_{cam_idx} {cam_name} fov: {cam_fovs[cam_idx]}")

    # case 查看 sensor 属性
    print("")
    n_sensor = model.nsensor
    sensor_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_idx)
        for sensor_idx in range(n_sensor)
    ]
    print(f"# n_sensor:  {n_sensor}")
    print(f"sensor_names: {sensor_names}")

    # case 查看 site 属性
    print("")
    n_site = model.nsite
    site_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_idx)
        for site_idx in range(n_site)
    ]
    print(f"# n_site: {n_site}")
    print(f"site_names: {site_names}")
    print(
        "-----------------------------------------------------------------------------"
    )


def init_viewer(model, data):
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(model, data)

            # with viewer.lock():
            #   viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2) # 可视化接触点
            #   viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = int(data.time % 2) # 可视化皮肤

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


def main():
    # ? logging.info 和 print 的区别？
    logging.info(
        colored(
            "###### Starting the simulation! ######",
            "yellow",
            attrs=["bold"],
        )
    )

    argparser = argparse.ArgumentParser

    # xml 模型路径
    xml_path = "src/UniRobot/sim/mujoco_env/assets/robots/tcb610/fujqr04_urdf.xml"

    # 加载模型
    model, data = load_model(xml_path)

    # 打印模型各属性
    print_info(model)

    # case1 重置到 "home" 关键帧（索引 0）
    mujoco.mj_resetData(model, data)  # Reset state and time.
    # mujoco.mj_resetDataKeyframe(model, data, 0)

    init_viewer(model, data)


if __name__ == "__main__":
    main()
