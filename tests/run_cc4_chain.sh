#!/bin/bash
# Wait for the 8-rank E2E (selmega_e2e.log) to finish, then the single-GPU reproducibility batch on GPU 4:
# 10 processes x 4 cells x 100 launches, back-to-back (gap=clone) and host-gap (gap=compute) regimes, plus the
# keys-only FE (DG_FE_SELECT_IN_MEGA=1, gap=clone). Logs in /raid/kimi/results/fe_cc4/.
cd "$(dirname "$0")/.."
R=/raid/kimi/results/fe_cc4
for i in $(seq 1 200); do grep -q "E2E_DONE" $R/selmega_e2e.log 2>/dev/null && break; sleep 15; done
export CUDA_VISIBLE_DEVICES=4
bash tests/run_fe_repro_cc.sh 10 100 B clone   > $R/repro_B_clone.log 2>&1
bash tests/run_fe_repro_cc.sh 10 100 B compute > $R/repro_B_compute.log 2>&1
DG_FE_SELECT_IN_MEGA=1 bash tests/run_fe_repro_cc.sh 10 100 B clone > $R/repro_B_selmega_clone.log 2>&1
echo "CHAIN_DONE $(date +%T)" >> $R/repro_B_selmega_clone.log
