ARG VERSION=v0.6.4.post1
FROM vllm/vllm-openai:${VERSION}
ARG VERSION

WORKDIR /root/adaptive-gpu-allocator/
RUN python3 -m pip install -U protobuf
RUN pip install ai-computing-broker

COPY ./cookbooks/vllm/adaptive-gpu-allocator/app/vllm/ ./app/vllm/
RUN chmod +x ./app/vllm/apply_patch.sh
RUN ./app/vllm/apply_patch.sh

WORKDIR /vllm-workspace

# Prepare the cache directory for vllm
RUN mkdir -p .cache/vllm
# Change the permissions so that all users can access here
RUN chmod -R 777 .

ENTRYPOINT ["agarun", "--", "python3", "-m", "vllm.entrypoints.openai.api_server"]
