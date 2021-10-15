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

cp ./object/train.py .

python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/train.py \
--src-dset-path './data/semi_processed_bcc' \
--labeled_num 2500 \
--num_classes 2 \
--suffix 'tc_bcc'

python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/train.py \
--src-dset-path './data/semi_processed_mel' \
--labeled_num 2500 \
--num_classes 2 \
--suffix 'tc_mel'

run_check 2500_two_cate
