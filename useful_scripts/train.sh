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
--max_epoch 60 \
--labeled_num 1000 \
--threshold 0.99 \
--weight-afm 0.5 \
--weight-u 0.5 \
--num_classes 7 \
--suffix 'neg_ce_dropout_1000'

run_check neg_1000
