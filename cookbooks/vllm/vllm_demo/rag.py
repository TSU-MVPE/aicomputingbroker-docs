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


import datetime
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.openai_like import OpenAILike

from .prompt_template import PromptTemplates


class RAGSystem:
    def __init__(self, servers=None):
        """
        Initialize RAG System with server configurations

        Args:
            servers: List of server configurations. If None, uses default configuration.
        """
        if servers is None:
            self.servers = [
                {
                    "addr": "localhost",
                    "model": "microsoft/Phi-4",
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
        else:
            self.servers = servers

        # Cache for loaded contexts and query engines
        self._context_cache = {}
        self._query_engine_cache = {}

        self.prompt_template = PromptTemplates.get_chat_qa_prompt()
        self.refine_prompt = PromptTemplates.get_chat_refine_prompt()

    def load_context(self, persist_dir):
        """Load storage context with caching"""
        if persist_dir in self._context_cache:
            return self._context_cache[persist_dir]

        cache_name = f"{persist_dir}.pickle"

        if os.path.exists(cache_name):
            print("load from persist_dir cache")
            with open(cache_name, "rb") as f:
                context = pickle.load(f)
        else:
            print("load from persist_dir (needs long time)")
            context = StorageContext.from_defaults(persist_dir=persist_dir)

            with open(cache_name, "wb") as f:
                pickle.dump(context, f)

        self._context_cache[persist_dir] = context
        return context

    def get_llm(self, server):
        """Create LLM instance for given server configuration"""
        return OpenAILike(
            api_base=f'http://{server["addr"]}:{server["port"]}/v1',
            api_key="EMPTY",
            model=server["model"],
            max_tokens=512,
            temperature=0.0,
            context_window=1024 * 7,
            is_chat_model=True,
        )

    def get_query_engine(self, server):
        """Create query engine for given server configuration with caching"""
        model_key = server["model"]

        if model_key in self._query_engine_cache:
            return self._query_engine_cache[model_key]

        llm = self.get_llm(server)

        embed_model_name = "bge-m3"
        embed_model = OllamaEmbedding(
            model_name=embed_model_name, base_url=f'http://{server["ollama_addr"]}:{server["ollama_port"]}'
        )

        context = self.load_context(server["storage_context"])
        index = load_index_from_storage(context, embed_model=embed_model)

        query_engine = index.as_query_engine(
            similarity_top_k=2,
            llm=llm,
            text_qa_template=self.prompt_template,
            refine_template=self.refine_prompt,
            streaming=True,
        )

        self._query_engine_cache[model_key] = query_engine
        return query_engine

    def _process_single_request(self, index, request):
        """Process a single request (internal method)"""
        model = request["model"]
        use_rag = request["use_rag"]
        server = next((server for server in self.servers if server["model"] == model), None)

        if server is None:
            return [
                {"time": datetime.datetime.now(), "user": f"u{index}", "model": model, "response": "(model not found)"}
            ]

        results = []
        if use_rag:
            results.append(
                {
                    "time": datetime.datetime.now(),
                    "user": f"u{index}",
                    "model": model,
                    "response": "(fetching indexed data)",
                }
            )
            query_engine = self.get_query_engine(server)

            response = query_engine.query(request["prompt"])

            response_gen = response.response_gen
            response_first = response_gen.__next__()

            response_full = response_first
            for chunk in response_gen:
                response_full += chunk
        else:
            results.append(
                {"time": datetime.datetime.now(), "user": f"u{index}", "model": model, "response": "(calling LLM)"}
            )
            llm = self.get_llm(server)
            response_full = str(llm.complete(request["prompt"]))

        t = datetime.datetime.now()

        results.append({"time": t, "user": f"u{index}", "model": model, "response": response_full})
        return results

    def process_requests(self, batched_requests):
        """
        Process multiple requests in parallel

        Args:
            batched_requests: List of request dictionaries

        Returns:
            List of response dictionaries sorted by time
        """
        with ThreadPoolExecutor() as executor:
            nested_results = list(
                executor.map(lambda item: self._process_single_request(*item), enumerate(batched_requests))
            )
            results = [item for sublist in nested_results for item in sublist]
        sorted_results = sorted(results, key=lambda x: x["time"])
        return sorted_results

    def query_single(self, model_name, prompt, use_rag=True):
        """
        Query a single model

        Args:
            model_name: Name of the model to query
            prompt: Query prompt
            use_rag: Whether to use RAG or not

        Returns:
            Response string
        """
        server = next((server for server in self.servers if server["model"] == model_name), None)
        if server is None:
            return "(model not found)"

        if use_rag:
            query_engine = self.get_query_engine(server)
            response = query_engine.query(prompt)

            response_gen = response.response_gen
            response_first = response_gen.__next__()

            response_full = response_first
            for chunk in response_gen:
                response_full += chunk
            return response_full
        else:
            llm = self.get_llm(server)
            return str(llm.complete(prompt))
