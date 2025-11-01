from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union
import yaml


@dataclass
class Config:
    experiment_name: str
    data: Dict[str, Any]
    training: Dict[str, Any]
    model: Dict[str, Any]
    logging: Dict[str, Any]
    evaluation: Dict[str, Any]

    @classmethod
    def from_file(cls, cfg_path: Union[str, Path]) -> "Config":
        with open(cfg_path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        return cls(**raw)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "data": self.data,
            "training": self.training,
            "model": self.model,
            "logging": self.logging,
            "evaluation": self.evaluation,
        }

    def update(self, overrides: Dict[str, Any]) -> None:
        for key, value in overrides.items():
            if hasattr(self, key) and isinstance(value, dict):
                getattr(self, key).update(value)
            else:
                setattr(self, key, value)
