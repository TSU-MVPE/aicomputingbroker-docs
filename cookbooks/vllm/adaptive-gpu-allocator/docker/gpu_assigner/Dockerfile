FROM ubuntu:22.04

RUN apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         python3 \
         python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists
COPY ./ /root/adaptive-gpu-allocator/
COPY ./license.lic /root/adaptive-gpu-allocator/license.lic
WORKDIR /root/adaptive-gpu-allocator/
RUN pip install ai-computing-broker

EXPOSE 11234
CMD gpu_assigner --address 0.0.0.0 start && sleep infinity
