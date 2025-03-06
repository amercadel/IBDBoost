#!/bin/bash
hap_input=$1
gt_segments_input=$2
cm_cutoff=$3
false_ibd_cutoff=$4

python findFalseIBD.py $hap_input $gt_segments_input $cm_cutoff $false_ibd_cutoff
python calculateStatistics.py $hap_input $gt_segments_input $cm_cutoff
