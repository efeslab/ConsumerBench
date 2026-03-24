import sys
sys.path.insert(0, ".")

from applications.OSTool.OSTool import OsTool

app = OsTool()
app.config = {
    "tool_kwargs": {
        "tool": "bash",
        "num_runs": 1,
        "data_file": "applications/OSTool/data/data_dental.json",
    }
}

app.run_setup()
result = app.run_application()
print(result)
app.run_cleanup()