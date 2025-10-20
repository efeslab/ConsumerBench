import json
import sys
import os
import re
import yaml

def remove_config_comments(file_path) -> str:
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        # Remove comments and empty lines
        cleaned_lines = [line for line in lines if not line.strip().startswith('#') and line.strip() != '']
        
        return ''.join(cleaned_lines)

def load_yaml(yaml_file):
    # Load the YAML file and remove comments
    cleaned_yaml = remove_config_comments(yaml_file)
    return yaml.safe_load(cleaned_yaml)    

def convert_yml_to_json(config_file_path):
    """
    Read a configuration file and convert it to JSON format.
    
    Args:
        config_file_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration as a JSON-compatible dictionary
    """
    # Check if file exists
    if not os.path.isfile(config_file_path):
        print(f"Error: File '{config_file_path}' not found.")
        sys.exit(1)

    # Read the config file
    try:
        with open(config_file_path, 'r') as file:
            config_text = file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    workflow_config = load_yaml(config_file_path)
    return workflow_config

def main():
    # Check if a file path was provided
    if len(sys.argv) < 5:
        print("Usage: python yml_to_json.py <config_file_path> <RESULTS_BASE_DIR> <CUR_DAY> <CUR_TIME>")
        sys.exit(1)
    
    # Get file paths from command line arguments
    yml_file_path = sys.argv[1]
    
    # Determine output file path
    if len(sys.argv) > 4:
        results_dir_base = sys.argv[2]
        results_cur_day = sys.argv[3]
        results_cur_time = sys.argv[4]
    else:
        # Default: use same name as input file but with .json extension
        output_file_path = os.path.splitext(yml_file_path)[0] + '.json'
    
    # Convert to JSON
    config_json = convert_yml_to_json(yml_file_path)
    
    # Output pretty JSON
    json_output = json.dumps(config_json, indent=4)

    # Get each application name and its device in a string
    results_dir = ""
    base_filename = yml_file_path.split("/")[-1].split(".")[0]
    results_dir = f"{base_filename}_"

    output_file_path = os.path.join(results_dir_base, results_cur_day, f"{results_dir}{results_cur_time}", "config.json")
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    # Save to file
    # print(f"Saving JSON configuration to '{output_file_path}'")
    try:
        with open(output_file_path, 'w') as f:
            f.write(json_output)
        # print(f"JSON configuration saved to '{output_file_path}'")
    except Exception as e:
        print(f"Error writing to output file: {e}")
        sys.exit(1)
    
    # Also print to console
    # print("\nGenerated JSON:")
    # print(json_output)

    return results_dir

if __name__ == "__main__":
    results_dir = main()
    print(f"{results_dir}")
