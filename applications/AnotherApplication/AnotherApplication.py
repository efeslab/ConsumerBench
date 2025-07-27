from typing import Any, Dict
import os
import sys

repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_dir)

from applications.application import Application

class AnotherApplication(Application):
    def __init__(self):
        super().__init__()

    def run_setup(self, *args, **kwargs):
        print("AnotherApplication setup")
        return {"status": "setup_complete", "config": self.config}

    def run_cleanup(self, *args, **kwargs):
        print("AnotherApplication cleanup")
        return {"status": "cleanup_complete"}

    def run_application(self, *args, **kwargs):
        print("AnotherApplication run")
        return {"status": "run_complete", "config": self.config}

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "another_config": "another_value"
        }

    def load_dataset(self):
        print("AnotherApplication loading dataset")
        return {"status": "dataset_loaded"}