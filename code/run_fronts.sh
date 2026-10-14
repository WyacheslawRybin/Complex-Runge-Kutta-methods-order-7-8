#!/bin/bash
# E_min(B) fronts of CRK7 and CRK8 (front_E.py rewrites data/front_E.json per family, so run them one after another)
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
python front_E.py CRK7 7,7 Emin_77_5.json,Emin_77_5_polish.json 4 CRK7-opt,CRK7s7b,CRK7-B0 4 > front_CRK7.log 2>&1
python front_E.py CRK8 9,8 Emin_98_5.json,Emin_98_5_polish.json 3 CRK8-opt,CRK8-b,CRK8s9 4 > front_CRK8.log 2>&1
echo "fronts done $(date)" >> front_CRK8.log
