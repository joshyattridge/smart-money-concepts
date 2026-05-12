#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=========================================="
echo "🚀 ICT-AI Trading Bot Deployment Script"
echo "=========================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null
then
    echo "Python3 could not be found. Installing..."
    sudo apt update
    sudo apt install python3 python3-pip python3-venv tmux -y
fi

# Create Virtual Environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Ensure data and models directories exist
mkdir -p data
mkdir -p models

echo ""
echo "✅ Deployment Environment Setup Complete!"
echo ""
echo "Next Steps:"
echo "1. Edit config.json with your API Keys: nano config.json"
echo "2. Train the AI model (First time only): python train.py"
echo "3. Start the bot in the background using tmux:"
echo "   tmux new -s bot"
echo "   source venv/bin/activate && python main.py"
echo "   (Press Ctrl+B, then D to detach and leave it running)"
