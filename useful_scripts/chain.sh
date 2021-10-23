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

:<<comment
python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/train.py \
--src-dset-path './data/semi_processed' \
--labeled_num 500 \
--num_classes 7 \
--suffix 'sev_cates'

run_check 500_two_cate
sleep 5s

python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/train.py \
--src-dset-path './data/semi_processed' \
--labeled_num 1000 \
--num_classes 7 \
--suffix 'sev_cates'

run_check 1000_two_cate
sleep 5s

python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/train.py \
--src-dset-path './data/semi_processed' \
--labeled_num 1500 \
--num_classes 7 \
--suffix 'sev_cates'

run_check 1500_two_cate
sleep 5s

python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/train.py \
--src-dset-path './data/semi_processed' \
--labeled_num 2000 \
--num_classes 7 \
--suffix 'sev_cates'

run_check 2000_two_cate
sleep 5s
comment

python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/object/train.py \
--src-dset-path './data/semi_processed' \
--labeled_num 2500 \
--num_classes 7 \
--suffix 'mc_base'

run_check 2500_two_cate
