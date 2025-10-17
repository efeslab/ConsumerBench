import time
from typing import Any, Dict
import sys
import os

repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_dir)

from applications.application import Application
import applications.MCPServer.mcp_manager as mcp_manager

class MCPServer(Application):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mcp_trace_file = kwargs.get('mcp_trace_file', None)
        self.config_file = kwargs.get('config_file', None)
        if self.config_file is not None and self.mcp_trace_file is not None:
            mcp_manager.load_mcp_trace_file(self.mcp_trace_file)
            mcp_manager.load_mcp_servers(self.config_file)
        self.mcp_ids = []

        
    def run_setup(self, *args, **kwargs):
        print("MCPServer setup")
        self.mcp_ids = kwargs.get('ids', self.get_default_config()['mcp_ids'])

        return {"status": "setup_complete", "config": self.config}    

    def run_cleanup(self, *args, **kwargs):
        print("MCPServer cleanup")
        return {"status": "cleanup_complete"}

    def run_application(self, *args, **kwargs):
        print(f"Running MCPServer")
        mcp_manager.run_mcp_server(mcp_ids=self.mcp_ids, logger=kwargs.get('logger', None))
        return {"status": "run_complete"}

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "type": "MCPServer",
            "mcp_ids": ["mcp_1"],
        }
    
    def load_dataset(self, *args, **kwargs):        
        return {"status": "dataset_loaded"}