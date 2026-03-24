import importlib
import inspect
import json
import os
import sys
import time
import sqlalchemy
from typing import Any, Callable, Dict, List
from .Interaction import Container

repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_dir)

from applications.application import Application

class DbBench(Application):
    """
    Application to benchmark database query performance.
    Config should include:
      - db_connection_string: str, connection string for the database
      - query: str, the SQL query to execute
      - num_runs: int, how many times to run the query for benchmarking
    """

    def __init__(self):
        super().__init__()
        self.tool = None
        self.query = None
        self.num_runs = 1
        self.container = None
        self.dataset = []
    
    # def _call_tool(self, tool: Any, tool_args: List[Any], tool_kwargs: Dict[str, Any]):
    #     if callable(tool):
    #         return tool(*tool_args, **tool_kwargs)
    #     if hasattr(tool, "forward") and callable(tool.forward):
    #         return tool.forward(*tool_args, **tool_kwargs)
    #     raise TypeError("Resolved tool is not callable and has no forward(...) method.")

    def run_setup(self, *args, **kwargs):
        print("DbBench setup")
        tool_kwargs = self.config.get("tool_kwargs", self.config)

        self.tool = kwargs.get("tool", tool_kwargs.get("tool"))
        self.query = kwargs.get("query", tool_kwargs.get("query"))
        self.num_runs = kwargs.get("num_runs", tool_kwargs.get("num_runs", 1))
        data_file = kwargs.get("data_file", tool_kwargs.get("data_file"))

        with open(data_file) as f:
            if data_file.endswith("json"):
                data = json.loads(f.read())
            else:
                data = [json.loads(line) for line in f.readlines()]
        print("data file read")
        for entry in data:
            if entry["type"][0] in ("INSERT", "DELETE", "UPDATE"):
                ans = entry.pop("answer_md5")
            else:
                ans = entry.pop("label")
            inp = entry
            self.dataset.append((inp, ans))

        self.container = Container()

        return {"status": "setup_complete", "config": self.config}

    def run_cleanup(self, *args, **kwargs):
        print("DbBench tool cleanup")
        self.tool = None

        # remove the database 
        container = self.container
        # if entry["type"][0] in ("INSERT", "DELETE", "UPDATE"):
        #     columns = ",".join(
        #         [
        #             f"`{column['name']}`"
        #             for column in entry["table"]["table_info"]["columns"]
        #         ]
        #     )
        #     md5_query = (
        #         f"select md5(group_concat(rowhash order by rowhash)) as hash "
        #         f"from( SELECT substring(MD5(CONCAT_WS(',', {columns})), 1, 5) AS rowhash FROM `{db}`) as sub;"
        #     )
        #     answer = container.execute(md5_query, db)
        if not self.dataset:
            raise RuntimeError("Dataset not loaded")
        entry = self.dataset[0][0]
        db = entry["create"]["database"]
        container.execute(f"drop database `{db}`")
        self.container.delete()
        return {"status": "cleanup_complete"}
    
    def run_application(self, *args, **kwargs):
        if not self.dataset:
            self.run_setup(*args, **kwargs)

        # set up the db
        entry = self.dataset[0][0]
        container = self.container
        print(entry)

        init_sql, init_data = self.build_init_sql(entry)
        container.execute(init_sql, data=init_data)
        print("We are done executing initial SQL here ----------")
        print(container.execute("SHOW DATABASES"))

        table = entry["table"]["table_name"]
        db = entry["create"]["database"]
        print(f"Database is {db}")
        response = container.execute(self.query, db)
        print(container.execute("SHOW DATABASES"))
        print(f"Response is: {response} ----------" )
        # avg_time = sum(times) / len(times)
        # print(f"Average execution time over {self.num_runs} runs: {avg_time:.4f} seconds")
    
    def load_dataset(self, *args, **kwargs):
        print("DbBench tool loading dataset")
        return {"status": "dataset_loaded"}
    
    def build_init_sql(self, entry):
        db = entry["create"]["database"]
        table = entry["table"]["table_name"]
        columns = ",".join(
            [
                f"`{column['name']}` TEXT"
                for column in entry["table"]["table_info"]["columns"]
            ]
        )
        column_names = ",".join(
            [f"`{column['name']}`" for column in entry["table"]["table_info"]["columns"]]
        )
        items = []
        items_data = ()
        for row in entry["table"]["table_info"]["rows"]:
            item = "("
            for col in row:
                item += "%s,"
                items_data += (col,)
            item = item[:-1] + ")"
            items.append(item)
        items = ",".join(items)
        sql = f"""CREATE DATABASE IF NOT EXISTS `{db}`;
            USE `{db}`;
            CREATE TABLE IF NOT EXISTS `{table}` ({columns});
            INSERT INTO `{table}` ({column_names}) VALUES {items}; 
            COMMIT;
            """
        return sql, items_data