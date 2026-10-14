#!/bin/bash
# CCK8 searches restarted after the interruption at 20:55 (casc98_8 r22 had 3362/6000 starts, 0 hits)
cd "$(dirname "$0")"
X=../../code/csearch
export OMP_NUM_THREADS=4
run() { echo "=== $(date +%H:%M:%S) $*" >> queue3.log; "$@" 2> /dev/null >> queue3.log; }
run $X/casc98_8.exe  2700 23 37:17,85:37,200:85 2000,2000,4000 h98_87_r23.bin
run $X/casc98_8.exe  2000 42 85:85,200:85 2000,4000 h98_87_seed42.bin seeds_98_87.bin 5172 0.05
run $X/casc108_9.exe 6000 32 37:17,85:37,200:85 2000,2000,4000 h108_97_r32.bin
echo "=== $(date +%H:%M:%S) done" >> queue3.log
