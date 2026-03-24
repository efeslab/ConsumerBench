import os
import sys
repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_dir)

from applications.DbBench.DbBench import DbBench

config = {
    "tool": "sql",
    "query": "UPDATE unknown SET Pos = 'FW', Name = 'Yuji Senuma' WHERE No = '20';",
    "num_runs": 1,
    "data_file": "applications/DbBench/data_dbbench.jsonl"
}

bench = DbBench()
bench.config = config

bench.run_setup()

bench.run_application()

bench.run_cleanup()