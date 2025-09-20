#!/bin/bash

# Script to easily run Docker commands for MNIST training

case "$1" in
    "build")
        echo "Building Docker image..."
        docker-compose build mnist-trainer
        ;;
    "train")
        echo "Running MNIST training..."
        docker-compose up mnist-trainer
        ;;
    "dev")
        echo "Starting development container..."
        docker-compose up -d mnist-dev
        echo "Container started. Access it with: docker exec -it mnist-dev bash"
        ;;
    "gpu")
        echo "Running MNIST training with GPU support..."
        docker-compose up mnist-gpu
        ;;
    "shell")
        echo "Opening shell in development container..."
        docker exec -it mnist-dev bash
        ;;
    "stop")
        echo "Stopping all containers..."
        docker-compose down
        ;;
    "clean")
        echo "Cleaning up containers and images..."
        docker-compose down --rmi all --volumes
        ;;
    *)
        echo "Usage: $0 {build|train|dev|gpu|shell|stop|clean}"
        echo ""
        echo "Commands:"
        echo "  build  - Build the Docker image"
        echo "  train  - Run training in container"
        echo "  dev    - Start development container"
        echo "  gpu    - Run training with GPU support"
        echo "  shell  - Open shell in development container"
        echo "  stop   - Stop all containers"
        echo "  clean  - Remove containers and images"
        exit 1
        ;;
esac