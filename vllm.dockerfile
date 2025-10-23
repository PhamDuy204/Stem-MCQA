FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Cài các gói cần thiết
RUN apt update && apt install -y git curl cmake build-essential

# Cài requirements
COPY ./requirements.txt .
RUN pip install -U bitsandbytes
RUN pip install -U vllm huggingface_hub --no-cache-dir
RUN mkdir -p ~/.cache/huggingface/
COPY ./llama3-1b-mcqa/checkpoint-6372 ./lora_module/checkpoint-6372
# Start command
CMD hf auth login --token ${HF_Token} && vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --api-key 'test' \
  --compilation-config '{"cache_dir": "/model_cache"}' \
  --port 8000 \
  --quantization bitsandbytes \
  --enable-prefix-caching \
  --swap-space 16 \
  --max-lora-rank 64 \
  --gpu-memory-utilization 0.8 \
  --disable-log-requests \
  --enable-sleep-mode \
  --max-model-len 8192 \
  --enable-lora \
  --lora-modules "my_lora=./lora_module/checkpoint-6372"

