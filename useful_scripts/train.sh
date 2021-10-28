#!/bin/bash
function run_check() {
	status=$?

	if [ $status != 0 ]; then
		echo "run $1-experiment failed"
		exit
	else
		echo "run $1-experiment successfully, come to the next experiment"
	fi
}

exp_flag='neg_ul_unlbl_ce_lbl'
python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/object/train.py \
--src-dset-path './data/semi_processed' \
--start_u 6 \
--max_epoch 60 \
--check_points_path 'ckps_ncpl_bl' \
--labeled_num 1000 \
--threshold 0.99 \
--weight-afm 0.5 \
--weight-u 0.5 \
--num_classes 7 \
--suffix $exp_flag

run_check $exp_flag
