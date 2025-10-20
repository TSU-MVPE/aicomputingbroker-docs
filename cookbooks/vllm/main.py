# Copyright 2025 Fujitsu Research of America, Inc.
#
# This software is licensed under an End User License Agreement (EULA) by Fujitsu
# Research of America, Inc. You are not allowed to use, copy, modify, or distribute
# this software and its documentation without express permission from Fujitsu Research
# of America, Inc. Please refer to the full EULA provided with this software for
# detailed information on permitted uses and restrictions.
#
# The software is provided "as is", without warranty of any kind, express or implied,
# including but not limited to the warranties of merchantability, fitness for a
# particular purpose and noninfringement. In no event shall Fujitsu Research of America,
# Inc. be liable for any claim, damages or other liability, whether in an action of
# contract, tort or otherwise, arising from, out of or in connection with the software
# or the use or other dealings in the software.

from vllm_demo.rag import RAGSystem


def main():
    # Default servers configuration
    servers = [
        {
            "addr": "localhost",
            # "model": "CohereLabs/c4ai-command-r-v01",  # 1 80GB GPU
            "model": "microsoft/Phi-4",  # 2 48GB GPUs
            "port": 8100,
            "ollama_addr": "localhost",
            "ollama_port": 11434,
            "storage_context": "./context/en",
        },
        {
            "addr": "localhost",
            "model": "SakanaAI/TinySwallow-1.5B-Instruct",
            "port": 8101,
            "ollama_addr": "localhost",
            "ollama_port": 11435,
            "storage_context": "./context/jp",
        },
        {
            "addr": "localhost",
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "port": 8102,
            "ollama_addr": "localhost",
            "ollama_port": 11436,
            "storage_context": "./context/en",
        },
    ]

    """Example usage with class"""
    rag_system = RAGSystem(servers=servers)

    # Batch query example (same as original usage)
    u0 = {
        "model": "SakanaAI/TinySwallow-1.5B-Instruct",
        "prompt": "ACBとは何ですか？",
        "use_rag": False,
    }
    u1 = {
        # "model": "CohereLabs/c4ai-command-r-v01",  # 1 80GB GPU
        "model": "microsoft/Phi-4",  # 2 48GB GPUs
        "prompt": "What is ACB?",
        "use_rag": False,
    }
    u2 = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "prompt": "What is ACB?",
        "use_rag": False,
    }

    model = [u0, u1, u2]
    list_responses = rag_system.process_requests(model)

    for response in list_responses:
        print(f'{response["time"]}: {response["user"]}: {response["model"]}: {response["response"]}')


if __name__ == "__main__":
    main()
