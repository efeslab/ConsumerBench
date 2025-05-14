import json
import sys
import os
import re

def convert_config_to_json(config_file_path):
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
    
    # Dictionary to hold the final JSON structure
    config_json = {}
    
    # Split by empty lines to get application blocks
    app_blocks = re.split(r'\n\s*\n', config_text.strip())
    
    for block in app_blocks:
        if not block.strip():
            continue
            
        # Extract the application name and its settings
        lines = block.strip().split('\n')
        if not lines:
            continue
            
        # First line should be the app name followed by a colon
        app_match = re.match(r'^([^:]+):\s*$', lines[0])
        if not app_match:
            print(f"Warning: Could not parse app name from line: '{lines[0]}'")
            continue

        if lines[0].startswith('#'):
            # Skip comments
            continue 
               
        app_name = app_match.group(1).strip()
        config_json[app_name] = {}
        
        # Process all remaining lines for this application
        for i in range(1, len(lines)):
            line = lines[i].strip()
            
            if not line or line.isspace():
                continue
                
            # Skip comments (lines starting with #)
            if line.startswith('#'):
                continue
                
            # Check for key-value pairs
            key_value_match = re.match(r'^([^=]+?)\s*=\s*(.+)$', line)
            if key_value_match:
                key = key_value_match.group(1).strip()
                value = key_value_match.group(2).strip()
                
                # Convert values to appropriate types
                if value.isdigit():
                    value = int(value)
                elif value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                    
                config_json[app_name][key] = value
    
    return config_json

def main():
    # Check if a file path was provided
    if len(sys.argv) < 5:
        print("Usage: python config_to_json.py <config_file_path> <RESULTS_BASE_DIR> <CUR_DAY> <CUR_TIME>")
        sys.exit(1)
    
    # Get file paths from command line arguments
    config_file_path = sys.argv[1]
    
    # Determine output file path
    if len(sys.argv) > 4:
        results_dir_base = sys.argv[2]
        results_cur_day = sys.argv[3]
        results_cur_time = sys.argv[4]
    else:
        # Default: use same name as input file but with .json extension
        output_file_path = os.path.splitext(config_file_path)[0] + '.json'
    
    # Convert to JSON
    config_json = convert_config_to_json(config_file_path)
    
    # Output pretty JSON
    json_output = json.dumps(config_json, indent=4)

    # Get each application name and its device in a string
    results_dir = ""
    for app_name, settings in config_json.items():
        device = settings.get('device', 'default')
        results_dir += f"{app_name}_{device}_"

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
