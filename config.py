import os
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str = "uni-wbc-vla"
    vision_model: str = "vit"
    language_model: str = "llama"
    action_dim: int = 11  # 6DOF机械臂 + 2DOF底盘 + 3DOF夹爪
    hidden_dim: int = 512
    num_layers: int = 6
    dropout: float = 0.1

@dataclass
class SimulationConfig:
    """仿真配置"""
    env_name: str = "dual_arm_mobile_manipulation"
    max_steps: int = 1000
    time_step: float = 0.02
    gravity: List[float] = (0, 0, -9.81)
    visualization: bool = True
    real_time_simulation: bool = True

@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 1000
    optimizer: str = "adamw"
    weight_decay: float = 1e-5
    gradient_clip: float = 10.0
    save_interval: int = 100
    log_interval: int = 10

@dataclass
class ExperimentConfig:
    """实验配置"""
    experiment_name: str = "default"
    experiment_dir: str = "results"
    log_dir: str = "logs"
    data_dir: str = "data"
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class RobotConfig:
    """机器人配置"""
    base_type: str = "omni"
    robot_type: str = "ur5"
    gripper_type: str = "robotiq"
    joint_limits: Dict[str, List[float]] = None
    velocity_limits: Dict[str, float] = None
    acceleration_limits: Dict[str, float] = None

class Config:
    """项目配置"""
    def __init__(self):
        self.model = ModelConfig()
        self.simulation = SimulationConfig()
        self.training = TrainingConfig()
        self.experiment = ExperimentConfig()
        self.robot = RobotConfig()

    def save(self, path: str = "config.json"):
        """保存配置到文件"""
        import json
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str = "config.json"):
        """从文件加载配置"""
        import json
        with open(path, "r") as f:
            data = json.load(f)
        config = cls()
        for key, value in data.items():
            setattr(config, key, value)
        return config

config = Config()