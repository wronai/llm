#!/bin/bash
# WronAI Local Weights & Biases Server Setup Script

# Setup local Weights & Biases server
echo "Setting up local Weights & Biases server..."
echo "This requires Docker and Docker Compose to be installed."

# Create directory for local W&B server
mkdir -p ../wandb-local

# Create docker-compose.yml file
cat > ../wandb-local/docker-compose.yml << 'EOF'
version: '3'
services:
  wandb:
    image: wandb/local:latest
    container_name: wandb-local
    ports:
      - "8080:8080"
    environment:
      - WANDB_USERNAME=admin
      - WANDB_PASSWORD=admin
    volumes:
      - ./data:/vol
EOF

echo "Local W&B server configuration created in wandb-local/docker-compose.yml"
echo "To start the server, run: cd wandb-local && docker-compose up -d"
echo "Then configure your training to use the local server with:"
echo "export WANDB_BASE_URL=http://localhost:8080"
echo "export WANDB_API_KEY=admin"

# Exit status
exit $?
