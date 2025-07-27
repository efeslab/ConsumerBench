from typing import Any, Dict
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