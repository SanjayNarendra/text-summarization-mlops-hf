from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    dataset_name: str
    raw_dataset_dir: Path
