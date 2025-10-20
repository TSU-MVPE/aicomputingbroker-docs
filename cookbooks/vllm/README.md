## AI Computing Broker with vLLM
This guide demonstrates how to deploy and run the AI Computing Broker (ACB) using the vLLM framework to host several LLMs on a single vLLM server.

(This document is also available at [TSU-MVPE/aicomputingbroker-docs](https://github.com/TSU-MVPE/aicomputingbroker-docs))

### Installation

1. Create a virtual environment:
```sh
python -m venv .venv
source .venv/bin/activate
```

2. Install AI Computing Broker:
see Quick Install section in the manual.

3. Install demo dependencies:
```bash
pip install llama-index==0.11.2 llama-index-embeddings-ollama llama-index-llms-openai-like
```

4. Setting Configuration:
Create an `.env` file under the docker folder:
```bash
# Proxy settings
http_proxy=
https_proxy=
HTTP_PROXY=
HTTPS_PROXY=
no_proxy=localhost,127.0.0.1
NO_PROXY=localhost,127.0.0.1

# ACB installation directory
ACB_DIR=/path/to/client/folder

# User permissions (run these commands to get values)
UID=XXXX  # Get from: id -u
GID=XXXX  # Get from: id -g

# Hugging Face cache directory
HF_HOME=/tmp/hf-cache
HUGGING_FACE_HUB_TOKEN=YOUR_HF_TOKEN

# GPU scheduler type
SCHEDULER=gpu-affinity
```

### Run vLLM Server with ACB
All required Docker containers are provided. For detailed instructions, refer to the Docker section in the user manual.

This demo demonstrates usage with `Phi-4`, `TinySwallow-1.5B-Instruct`, and `TinyLlama-1.1B-Chat-v1.0`, integrated with RAG-based data sources.

```bash
cd docker
docker compose up -d
```

### Simple Run:
```sh
python main.py
```
