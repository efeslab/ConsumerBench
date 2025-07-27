import time
from typing import Any, Dict
from applications.application import Application


class SleepApplication(Application):
    def __init__(self):
        super().__init__()

    def run_setup(self, *args, **kwargs):
        print("SleepApplication setup")
        return {"status": "setup_complete", "config": self.config}

    def run_cleanup(self, *args, **kwargs):
        print("SleepApplication cleanup")
        return {"status": "cleanup_complete"}

    def run_application(self, *args, **kwargs):
        sleep_time = kwargs.get("sleep_time", self.config.get("sleep_time", 1.0))
        print(f"SleepApplication sleeping for {sleep_time} seconds")
        time.sleep(sleep_time)
        return {"status": "sleep_complete", "sleep_time": sleep_time}

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "sleep_time": 1.0
        }
    