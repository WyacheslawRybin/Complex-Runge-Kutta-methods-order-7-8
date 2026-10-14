#!/bin/bash
# Cash-Karp searches, run sequentially; summaries appended to queue1.log
cd "$(dirname "$0")"
X=../../code/csearch
export OMP_NUM_THREADS=10
run() { echo "=== $(date +%H:%M:%S) $*" >> queue1.log; "$@" 2> /dev/null >> queue1.log; }
run $X/casc98_8.exe  480 21 37:17,85:37,200:85 2000,2000,4000 h98_87_t21.bin
run $X/casc108_9.exe 480 31 37:17,85:37,200:85 2000,2000,4000 h108_97_t31.bin
run $X/casc87_7.exe 6000 11 37:17,85:37 2000,4000 h87_76_s11.bin
run $X/casc77_0.exe 20000 101 37:0,85:0 2000,4000 h77_ctl_s101.bin
run $X/casc77_6.exe 40000 201 37:17,85:37 2000,4000 h77_66_s201.bin
echo "=== $(date +%H:%M:%S) done" >> queue1.log
