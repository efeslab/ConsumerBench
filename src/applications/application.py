from abc import ABC, abstractmethod
from typing import Any, Dict

class Application(ABC):
    def __init__(self):
        self.config = self.get_default_config()

    def get_default_config(self) -> Dict[str, Any]:
        return {}
    
    def get_custom_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        default_config = self.get_default_config()
        custom_config = {}
        for key, value in config.items():
            if key in default_config:
                custom_config[key] = value
        return custom_config
    
    def add_config(self, config: Dict[str, Any]):
        self.config.update(config)

    # have to return something
    @abstractmethod
    def run_setup(self, *args, **kwargs):
        pass

    # have to return something
    @abstractmethod
    def run_cleanup(self, *args, **kwargs):
        pass

    # have to return something
    @abstractmethod
    def run_application(self, *args, **kwargs):
        pass