# Container image for Awesome-Perception-Aware-RLVR.
#
# Base: the official vLLM 0.19.0 image (Ubuntu + CUDA 12.8 + torch 2.10 + vLLM 0.19), the
# stack the repository was tested with. flash-attn is compiled from source because no
# prebuilt wheel targets torch 2.10; set MAX_JOBS to match the build machine's memory.
#
#   docker build -t parlvr:latest .
#   docker run -it --gpus all --ipc=host -v $PWD:/workspace/parlvr parlvr:latest
FROM vllm/vllm-openai:v0.19.0

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_ROOT_USER_ACTION=ignore \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    MAX_JOBS=16

RUN apt-get update && apt-get install -y --no-install-recommends git tini && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/parlvr
COPY requirements.txt scripts/constraints.txt /tmp/
RUN grep -Ev '^(flash-attn|vllm)([<>=!~].*)?$' /tmp/requirements.txt > /tmp/requirements.runtime.txt && \
    pip install --no-cache-dir -r /tmp/requirements.runtime.txt -c /tmp/constraints.txt && \
    pip install --no-cache-dir ninja packaging && \
    pip install --no-cache-dir --no-build-isolation "flash-attn==2.8.3"

COPY . /workspace/parlvr
RUN pip install --no-cache-dir --no-deps -e .

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
