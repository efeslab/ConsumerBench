import re
import pandas as pd

file_path = "scripts/plots/concurrent/plot_data/livecaption-no-mps.txt"
with open(file_path, 'r') as f:
    log_text = f.read()

lines = log_text.strip().splitlines()[1:]  # skip header

# Parse into (name, timestamp) tuples
entries = []
for line in lines:
    parts = line.strip().split('\t')
    if len(parts) < 2:
        continue
    name = parts[0].strip()
    try:
        timestamp = float(parts[1].replace('s', ''))
    except ValueError:
        continue
    entries.append((name, timestamp))

# Process each request
requests = []
i = 0
while i < len(entries):
    name, timestamp = entries[i]
    if name.startswith("[Whisper request") and "START" in name:
        request_start = timestamp
        encode_time = 0.0
        last_encode_done = None
        last_model_generate_done = None
        i += 1
        while i < len(entries) and not entries[i][0].endswith("END]"):
            if entries[i][0] == "encode" and i+1 < len(entries) and entries[i+1][0] == "encode done":
                encode_start = entries[i][1]
                encode_end = entries[i+1][1]
                encode_time += encode_end - encode_start
                last_encode_done = encode_end
                i += 2
            elif entries[i][0] == "model.generate done":
                last_model_generate_done = entries[i][1]
                i += 1
            else:
                i += 1
        if i < len(entries) and entries[i][0].endswith("END]"):
            request_end = entries[i][1]
            latency = request_end - request_start
            # decode_time = (last_model_generate_done - last_encode_done) if last_model_generate_done and last_encode_done else 0
            decode_time = request_end - request_start - encode_time

            requests.append({
                "latency": latency,
                "encode": encode_time,
                "decode": decode_time
            })
            i += 1
    else:
        i += 1

# Create DataFrame
df = pd.DataFrame(requests)

# remove the max one
df = df[df["latency"] != df["latency"].max()]
# Compute summary
df.loc["average"] = df.mean()
df.loc["median"] = df.median()
df.loc["min"] = df.min()
df.loc["max"] = df.max()

print(f"Number of requests: {len(df) - 4}")  # exclude avg/median/min/max

# Print stats
print("Min latency:", df.loc["min", "latency"])
print("Max latency:", df.loc["max", "latency"])
print("Average latency:", df.loc["average", "latency"])
print("Median latency:", df.loc["median", "latency"])
print("Average encode ratio:", df.loc["average", "encode"] / df.loc["average", "latency"])
print("Median encode ratio:", df.loc["median", "encode"] / df.loc["median", "latency"])
print("Average decode ratio:", df.loc["average", "decode"] / df.loc["average", "latency"])
print("Median decode ratio:", df.loc["median", "decode"] / df.loc["median", "latency"])
