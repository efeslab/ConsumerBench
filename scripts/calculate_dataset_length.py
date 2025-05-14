from datasets import load_dataset
import json
import numpy as np

def calculate_length_for_two_roles(conversation_str):
    """
    Calculate the length of a conversation that has exactly two roles:
    first role is "user" and second role is "assistant".
    
    Args:
        conversation_str: JSON string representation of the conversation
        
    Returns:
        Dictionary with user_length, assistant_length, and total_length if valid
        None if the conversation doesn't meet the criteria
    """
    try:
        # Parse JSON string into a Python object
        # conversation = json.loads(conversation_str)
        conversation = conversation_str
        
        # Check if this is a two-turn conversation with user first, assistant second
        if (len(conversation) != 2 or
            conversation[0]["role"] != "user" or
            conversation[1]["role"] != "assistant"):
            return None
        
        # Calculate lengths
        user_length = len(conversation[0]["content"])
        assistant_length = len(conversation[1]["content"])
        total_length = user_length + assistant_length
        
        return {
            "user_length": user_length,
            "assistant_length": assistant_length,
            "total_length": total_length
        }
    except (json.JSONDecodeError, KeyError, TypeError):
        # Skip any invalid entries
        return None

def main():
    # Load the dataset (requires Hugging Face authentication)
    print("Loading the LMSYS Chat-1M dataset...")
    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("lmsys/lmsys-chat-1m")

    # Track maximum values
    max_total_length = 0
    max_user_length = 0
    max_assistant_length = 0
    max_index = -1

    # Store all valid total lengths for percentile calculations
    all_total_lengths = []

    print("Analyzing conversations...")
    processed_count = 0
    valid_count = 0

    # Process all conversations in the dataset
    for idx, conv_str in enumerate(ds["train"]["conversation"]):
        processed_count += 1
        
        # Print progress periodically
        if processed_count % 10000 == 0:
            print(f"Processed {processed_count} conversations...")
        
        # Calculate lengths for this conversation
        result = calculate_length_for_two_roles(conv_str)
        
        # If valid conversation with user first, assistant second
        if result:
            valid_count += 1
            
            # Add to list for percentile calculations
            all_total_lengths.append(result["total_length"])
            
            # Update max values if this is longer than previous max
            if result["total_length"] > max_total_length:
                max_total_length = result["total_length"]
                max_user_length = result["user_length"]
                max_assistant_length = result["assistant_length"]
                max_index = idx

    # Calculate percentiles
    percentile_90 = np.percentile(all_total_lengths, 90)
    percentile_99 = np.percentile(all_total_lengths, 99)

    # Print results
    print(f"\nTotal conversations processed: {processed_count}")
    print(f"Valid two-role conversations: {valid_count}")

    print(f"\nMaximum total length: {max_total_length}")
    print(f"- User message length: {max_user_length}")
    print(f"- Assistant message length: {max_assistant_length}")
    print(f"- Found at index: {max_index}")

    print(f"\nPercentiles:")
    print(f"- 90th percentile of total length: {percentile_90:.0f}")
    print(f"- 99th percentile of total length: {percentile_99:.0f}")

if __name__ == "__main__":
    main()