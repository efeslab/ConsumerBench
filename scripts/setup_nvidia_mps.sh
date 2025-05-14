#!/bin/bash
# NVIDIA MPS daemon service setup script

# Create a systemd service file for MPS
cat > /etc/systemd/system/nvidia-mps.service << 'EOF'
[Unit]
Description=NVIDIA Multi-Process Service Daemon
After=syslog.target

[Service]
Type=forking
User=root
Environment=CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
Environment=CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps
Environment=CUDA_VISIBLE_DEVICES=0
ExecStartPre=/bin/mkdir -p /var/log/nvidia-mps
ExecStartPre=/bin/mkdir -p /tmp/nvidia-mps
ExecStart=/usr/bin/nvidia-cuda-mps-control -d
ExecStop=/bin/bash -c "echo quit | /usr/bin/nvidia-cuda-mps-control"
Restart=on-abnormal

[Install]
WantedBy=multi-user.target
EOF

# Create directory for MPS logs if it doesn't exist
mkdir -p /var/log/nvidia-mps
mkdir -p /tmp/nvidia-mps

# Reload systemd, enable and start the service
systemctl daemon-reload
systemctl enable nvidia-mps.service
systemctl start nvidia-mps.service

echo "NVIDIA MPS daemon has been set up and started."
echo "Check status with: systemctl status nvidia-mps.service"