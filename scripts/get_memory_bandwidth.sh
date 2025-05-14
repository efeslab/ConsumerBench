#!/bin/bash

memory_throughput=$(sudo pcm-memory | grep "System Memory Throughput")
echo "Memory throughput: $memory_throughput MB/s"
