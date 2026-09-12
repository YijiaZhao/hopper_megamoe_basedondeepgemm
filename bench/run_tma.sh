#!/bin/bash
nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,utilization.gpu --format=csv,noheader -i 0
echo COMPUTE_APPS: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr "\n" " ")
docker exec four_api_build bash -lc "cd /raid/kimi && nvcc -arch=sm_90a -O3 -lcuda -o tma_microbench tma_microbench.cu 2>&1 && CUDA_VISIBLE_DEVICES=0 ./tma_microbench" 2>&1
