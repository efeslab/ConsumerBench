import asyncio
import os
import threading
from mcp_agent.mcp.mcp_server_registry import ServerRegistry
from mcp_agent.config import Settings, get_settings
from mcp_agent.core.context import Context, configure_logger
from mcp_agent.mcp.mcp_aggregator import MCPAggregator
from mcp_agent.logging.logger import get_logger
import yaml

mcp_server_names = []
mcp_tool_names = []
mcp_aggregator = None
mcp_trace = None
filesystem_tool_results_dir = os.path.dirname(os.path.abspath(__file__))

def start_background_event_loop():
    loop = asyncio.new_event_loop()

    def run_loop():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    t = threading.Thread(target=run_loop, daemon=True)
    t.start()
    return loop

mcp_server_loop = start_background_event_loop()

def get_tool_info_from_mcp_id(mcp_id):
    global mcp_trace
    for section_name, section_data in mcp_trace.items():
        if section_name == "tool_call":
            for call_id, call_data in section_data.items():
                if call_id == mcp_id:
                    return call_data.get('name', None), call_data.get('arguments', None)


def get_prompt_from_mcp_id(mcp_id):
    global mcp_trace
    for section_name, section_data in mcp_trace.items():
        if section_name == "text_generate":
            for call_id, call_data in section_data.items():
                if call_id == mcp_id:
                    return call_data.get('prompt', None)


def load_mcp_trace_file(trace_file):
    global mcp_trace
    with open(trace_file, 'r') as file:
        mcp_trace = yaml.safe_load(file)
    return mcp_trace


async def setup_mcp_servers(mcp_server_names):
    global mcp_aggregator
    global mcp_server_loop
    global filesystem_tool_results_dir
    
    if mcp_server_loop is None:
        mcp_server_loop = asyncio.get_running_loop()

    logger = get_logger(__name__)
    
    os.makedirs(filesystem_tool_results_dir, exist_ok=True)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    config = get_settings(f"{current_dir}/mcp_servers_config.yml")
    print(config)

    context = Context()
    context.config = config
    context.server_registry = ServerRegistry(config=config)

    await configure_logger(config, context.session_id)  

    if "filesystem" in context.config.mcp.servers:
        context.config.mcp.servers["filesystem"].args.extend([filesystem_tool_results_dir])
        logger.info("Filesystem server configured")


    mcp_aggregator = MCPAggregator(server_names=mcp_server_names, connection_persistence=True, context=context, name="test_aggregator")
    await mcp_aggregator.initialize()

    print(mcp_aggregator.initialized)
    return mcp_aggregator


async def run_mcp_server_async(**kwargs):
    global mcp_aggregator
    global filesystem_tool_results_dir

    mcp_ids = kwargs.get('mcp_ids', None)

    mcp_tool_name, arguments = get_tool_info_from_mcp_id(mcp_ids[0])
    logger = kwargs.get('logger', None)

    if "filesystem" in mcp_tool_name:
        if "path" in arguments:
            arguments["path"] = arguments["path"].replace("/workspace", filesystem_tool_results_dir)
        if "paths" in arguments:
            updated_paths = []
            for path in arguments["paths"]:
                updated_paths.append(path.replace("/workspace", filesystem_tool_results_dir))
            arguments["paths"] = updated_paths
        if "source" in arguments:
            arguments["source"] = arguments["source"].replace("/workspace", filesystem_tool_results_dir)
        if "destination" in arguments:
            arguments["destination"] = arguments["destination"].replace("/workspace", filesystem_tool_results_dir)

    print(f"Calling MCP server {mcp_tool_name} with arguments: {arguments}")

    # print(mcp_tool_name, arguments)
    res = await mcp_aggregator.call_tool(mcp_tool_name, arguments, None)
    if logger is not None:
        logger.info(f"MCP server {mcp_ids} result: {res.content}")
    else:
        print(res.content)

def run_mcp_server(**kwargs):
    # from mcp_manager import mcp_server_loop  # import global loop
    global mcp_server_loop
    
    if mcp_server_loop is None:
        raise RuntimeError("MCP server was not set up. Call load_mcp_servers() first.")
    
    # Submit async work to the correct loop
    future = asyncio.run_coroutine_threadsafe(run_mcp_server_async(**kwargs), mcp_server_loop)
    return future.result()

async def load_mcp_servers_async(config_path):
    global mcp_server_names
    global mcp_tool_names

    with open(config_path, 'r') as file:
        data = yaml.safe_load(file)
    
    tools = []
    
    # Iterate through all sections in the YAML
    for section_name, section_data in data.items():
        # Skip the workflows section and anything after it
        if section_name == 'workflows':
            break
            
        # Check if this section has type: mcp_server
        if isinstance(section_data, dict) and section_data.get('type') == 'MCPServer':
            tools.append(section_name)
            mcp_server_name = section_data.get('server_name')
            if mcp_server_name not in mcp_server_names:
                mcp_server_names.append(mcp_server_name)

    for tool in tools:
        tool_parts = tool.split('-')
        tool_parts = tool_parts[:-1]
        tool_name = ''
        for part in tool_parts:
            if tool_name != '':
                tool_name += '-' + part
            else:
                tool_name += part

        if tool_name not in mcp_tool_names:
            mcp_tool_names.append(tool_name)

    await setup_mcp_servers(mcp_server_names)

def load_mcp_servers(config_path):
    future = asyncio.run_coroutine_threadsafe(load_mcp_servers_async(config_path), mcp_server_loop)
    return future.result()
    