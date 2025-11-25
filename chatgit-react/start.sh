#!/bin/bash

# Start Backend
echo "Starting Backend..."
cd ../
source venv/bin/activate
uvicorn api:app --reload --port 8000 &
BACKEND_PID=$!

# Start Frontend
echo "Starting Frontend..."
cd chatgit-react/frontend
npm run dev -- --port 3000

# Cleanup on exit
kill $BACKEND_PID
