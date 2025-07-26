#!/bin/bash

# Check if port number is supplied as argument
if [ -z "$1" ]; then
    echo "Usage: $0 <port>"
    exit 1
fi

port=$1

# Validate if the input is a valid number
if ! [[ "$port" =~ ^[0-9]+$ ]]; then
    echo "Invalid port number."
    exit 1
fi

# Find the PID of the process using the specified port
pid=$(lsof -t -i :$port)

# If no process is found, notify the user
if [ -z "$pid" ]; then
    echo "No process is running on port $port."
    exit 0
fi

kill -2 $pid
echo "Process with PID $pid has been killed."
sleep 1
kill -9 $pid
