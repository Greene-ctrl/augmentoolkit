# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    redis-server \
    nodejs \
    npm \
    tesseract-ocr \
    poppler-utils \
    libmagic1 \
    gcc \
    python3-dev \
    cmake \
    build-essential \
    pkg-config \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_lg

# Clone and build llama.cpp
RUN git clone https://github.com/ggml-org/llama.cpp.git && \
    cd llama.cpp && \
    git checkout 3cb203c89f60483e349f841684173446ed23c28f && \
    cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF && \
    cmake --build build --config Release -j$(nproc)

# Copy the rest of the application code
COPY . .

# Build the frontend
WORKDIR /app/atk-interface
RUN npm install
RUN chmod +x node_modules/.bin/vite
RUN npm run build

# Back to root
WORKDIR /app

# Create necessary directories
RUN mkdir -p inputs outputs external_configs logs helper_process_logs models

# Download the model
RUN python download_model.py

# Create a startup script
RUN echo '#!/bin/bash\n\
redis-server --daemonize yes\n\
huey_consumer tasks.huey > helper_process_logs/huey.log 2>&1 &\n\
# Start LLM server in background\n\
python -c "import asyncio; from generation.utilities.llm_server.llm_server import llm_server; asyncio.run(llm_server(prompt_path=\"inputs/default_prompt.txt\", template_path=\"generation/core_pipelines/recall_multiple_sources/prompts/multi_turn_assistant_conversation.yaml\", gguf_model_path=\"models/augmentoolkit-v0.1/Augmentoolkit-DataSpecialist-7.2B-Q8_0.gguf\", context_length=4096, llama_path=\"./llama.cpp\", port=8003))" > helper_process_logs/llm_server.log 2>&1 &\n\
uvicorn api:app --host 0.0.0.0 --port 7860\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose the port the app runs on
EXPOSE 7860

# Set environment variables
ENV REDIS_HOST=localhost
ENV REDIS_PORT=6379

# Command to run the application
CMD ["/app/start.sh"]
