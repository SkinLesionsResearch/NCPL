#/bin/bash
function run_check() {
	status=$?

	if [ $status != 0 ]; then
		echo "run $1-experiment failed"
		exit
	else
		echo "run $1-experiment successfully, come to the next experiment"
	fi 
}

python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/train.py \
--src-dset-path './data/semi_processed_mel' \
--labeled_num 2500 \
--net "ale" \
--num_classes 2 \
--start_u 6 \
--threshold 0 \
--suffix '_tc_mel'

run_check tc_mel

python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/train.py \
--src-dset-path './data/semi_processed_bcc' \
--labeled_num 2500 \
--net "ale" \
--start_u 6 \
--num_classes 2 \
--threshold 0 \
--suffix '_tc_bcc'

run_check tc_bcc
