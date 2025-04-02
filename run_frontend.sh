#!/bin/bash

# Navigate to the frontend directory
cd frontend

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
  echo "Installing dependencies..."
  npm install
fi

# Start the development server with debugging enabled
echo "Starting frontend development server..."
REACT_APP_DEBUG=true npm start 