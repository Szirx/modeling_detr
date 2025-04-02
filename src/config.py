from omegaconf import OmegaConf
from pydantic import BaseModel


class DataConfig(BaseModel):
    data_path: str
    batch_size: int
    n_workers: int
    image_size: int
    processor_image_size: int

class ClearMLConfig(BaseModel):
    project_name: str
    task: str

class MlflowConfig(BaseModel):
    run_name: str
    experiment_name: str
    tracking_uri: str

class Config(BaseModel):
    data_config: DataConfig
    clearml_config: ClearMLConfig
    mlflow_config: MlflowConfig
    n_epochs: int
    num_queries: int
    num_classes: int
    accelerator: str
    devices: str
    monitor_metric: str
    model: str
    model_path: str
    ckpt_path: str
    id2label: dict
    processor: str
    optimizer: str
    optimizer_kwargs: dict
    scheduler: str
    scheduler_kwargs: dict
    threshold: float
    patience: int
    save_top_k: int
    log_every_n_steps: int

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)