import yaml
from pathlib import Path


class Config:
    path_to_config: Path = Path('config.yaml')
    @classmethod
    def load_config(cls):
        with open(cls.path_to_config, 'r') as file:
            cls._config = yaml.safe_load(file)
        cls.path_to_labels = cls._config['path_to_labels']
        cls.save_temporary_result_path = cls._config['save_temporary_result_path']
        cls.save_path = cls._config['save_path']


    @classmethod
    def set_path_to_config(cls, path: Path):
        cls.path_to_config = path


Config.load_config()