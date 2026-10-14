#!/bin/bash
# CCK8 searches (after queue1); summaries appended to queue2.log
cd "$(dirname "$0")"
X=../../code/csearch
while ! grep -q "=== .* done" queue1.log; do sleep 60; done
export OMP_NUM_THREADS=11
run() { echo "=== $(date +%H:%M:%S) $*" >> queue2.log; "$@" 2> /dev/null >> queue2.log; }
run $X/casc98_8.exe  6000 22 37:17,85:37,200:85 2000,2000,4000 h98_87_r22.bin
run $X/casc98_8.exe  2000 42 85:85,200:85 2000,4000 h98_87_seed42.bin seeds_98_87.bin 5172 0.05
run $X/casc108_9.exe 6000 32 37:17,85:37,200:85 2000,2000,4000 h108_97_r32.bin
echo "=== $(date +%H:%M:%S) done" >> queue2.log
