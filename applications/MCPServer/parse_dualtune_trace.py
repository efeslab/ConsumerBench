import json
import argparse
import os
import yaml
from collections import defaultdict

def extract_traces_and_build_dag(jsonl_path, output_dir):
    text_generations = {}
    tool_calls = {}
    node_order = []  # List of tuples: (type, id, extra_info)

    current_prompt = None
    text_id = 0
    tool_id = 0
    pending_tool = None

    tool_name_counter = defaultdict(int)

    os.makedirs(output_dir, exist_ok=True)

    trajs = [line.strip() for line in open(jsonl_path).readlines()]
    trace = []
    tools = []
    for traj in trajs:
        trace = json.loads(traj)

    tools = trace["tools"]
    messages = trace["messages"]
        
    llm_inputs = []
    llm_outputs = []    

    message_idx = 0
    llm_inputs.append(json.dumps(messages[0]))
    llm_outputs.append(json.dumps(messages[1]))
    gen_id = f"id_{text_id}"
    text_generations[gen_id] = {
        "prompt": llm_inputs[-1],
        "answer": llm_outputs[-1]
    }
    node_order.append(("text_generate", gen_id, None, None))
    text_id += 1

    for message in messages[:-1]:
        if message["role"] == "tool":
            for tool_call in messages[message_idx-1]["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])
                server_name = "filesystem"
                pending_tool = {
                    "name": f"filesystem_{tool_name}",
                    "arguments": tool_args,
                    "server_name": server_name
                }

            try:
                call_id = f"id_{tool_id}"
                tool_calls[call_id] = {
                    **pending_tool,
                    "answer": "\n".join([message["content"]])
                }
                # tool_name = pending_tool["name"].replace("_", "-")
                tool_name = pending_tool["name"]
                node_order.append(("tool_call", call_id, tool_name, pending_tool["server_name"]))
                tool_id += 1
                pending_tool = None
            except Exception as e:
                print(f"Warning: failed to extract tool response: {e}")

            llm_inputs.append(json.dumps(messages[:message_idx+1]))
            llm_outputs.append(json.dumps(messages[message_idx+1]))
            gen_id = f"id_{text_id}"
            text_generations[gen_id] = {
                "prompt": llm_inputs[-1],
                "answer": llm_outputs[-1]
            }
            node_order.append(("text_generate", gen_id, None, None))
            text_id += 1



        message_idx += 1

    # === Save JSON Trace ===
    trace_json_path = os.path.join(output_dir, "trace.json")
    with open(trace_json_path, 'w') as out_f:
        json.dump({
            "text_generate": text_generations,
            "tool_call": tool_calls
        }, out_f, indent=2)

    # === Build DAG and YAML ===
    yaml_dict = {}
    workflows = {}

    previous_step = None
    chatbot_counter = 1
    toolcall_counter = 1

    for op_type, op_id, tool_name, tool_server_name in node_order:
        if op_type == "text_generate":
            comp_name = f"chatbot{chatbot_counter}"
            workflows[f"generate_{chatbot_counter}"] = {
                "uses": comp_name
            }
            if previous_step:
                workflows[f"generate_{chatbot_counter}"]["depend_on"] = [previous_step]

            yaml_dict[comp_name] = {
                "server_model": "/mnt/storage/rohan/models/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-f16.gguf",
                "num_requests": 1,
                "ids": [op_id],
                "device": "gpu",
                "type": "Chatbot",
                "mps": 100
            }
            previous_step = f"generate_{chatbot_counter}"
            chatbot_counter += 1

        elif op_type == "tool_call":
            tool_name = f"{tool_name}"
            tool_server_name = tool_server_name
            tool_name_counter[tool_name] += 1
            comp_name = f"{tool_name}-{tool_name_counter[tool_name]}"

            workflows[f"tool_call_{toolcall_counter}"] = {
                "uses": comp_name
            }
            if previous_step:
                workflows[f"tool_call_{toolcall_counter}"]["depend_on"] = [previous_step]

            yaml_dict[comp_name] = {
                "num_requests": 1,
                "ids": [op_id],
                "server_name": tool_server_name,
                "type": "MCPServer"
            }
            previous_step = f"tool_call_{toolcall_counter}"
            toolcall_counter += 1

    yaml_dict["workflows"] = workflows

    # print(json.dumps(yaml_dict, indent=2))

    dag_yaml_path = os.path.join(output_dir, "dag.yaml")
    with open(dag_yaml_path, 'w') as f:
        yaml.dump(yaml_dict, f, sort_keys=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract traces and build DAG from MCP trace")
    parser.add_argument("-t", "--trace_file", type=str, required=True, help="Path to the MCP trace JSONL file")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Directory to store the output JSON and YAML")
    args = parser.parse_args()

    extract_traces_and_build_dag(args.trace_file, args.output_dir)
