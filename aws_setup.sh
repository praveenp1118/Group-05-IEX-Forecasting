#!/bin/bash
# AWS EC2 Setup Script for Group 05 IEX Forecasting
# Run this on EC2 after SSH in

echo "=== Setting up EC2 for IEX Forecasting ==="

# 1. Update system
sudo yum update -y

# 2. Install Docker
sudo yum install -y docker git
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ec2-user

# 3. Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
  -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

echo "Docker installed ✅"
docker --version
docker-compose --version
