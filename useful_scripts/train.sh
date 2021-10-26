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

python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/object/train.py \
--src-dset-path './data/semi_processed' \
--labeled_num 2500 \
--train_path 'ckps_neg' \
--threshold 0.99 \
--weight-afm 0.5 \
--weight-u 0.5 \
--num_classes 7 \
--suffix 'neg_2500'
run_check neg_2500
