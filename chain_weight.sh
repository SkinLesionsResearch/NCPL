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
--src-dset-path './data/semi_processed' \
--labeled_num 500 \
--num_classes 7 \
--weight-afm 0.5 \
--weight-u 0.5 \
--suffix 'sev_cates_weight'

run_check 5_5_sev_cate
sleep 5s

python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/train.py \
--src-dset-path './data/semi_processed' \
--labeled_num 500 \
--num_classes 7 \
--weight-afm 0.6 \
--weight-u 0.4 \
--suffix 'sev_cates_weight'

run_check 6_4_sev_cate
sleep 5s

python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/train.py \
--src-dset-path './data/semi_processed' \
--labeled_num 500 \
--num_classes 7 \
--weight-afm 0.7 \
--weight-u 0.3 \
--suffix 'sev_cates_weight'

run_check 7_3_sev_cate
sleep 5s

python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/train.py \
--src-dset-path './data/semi_processed' \
--labeled_num 500 \
--num_classes 7 \
--weight-afm 0.8 \
--weight-u 0.2 \
--suffix 'sev_cates_weight'

run_check 8_2_sev_cate
sleep 5s

python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/train.py \
--src-dset-path './data/semi_processed' \
--labeled_num 500 \
--num_classes 7 \
--weight-afm 0.9 \
--weight-u 0.1 \
--suffix 'sev_cates_weight'

run_check 9_1_sev_cate

python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/train.py \
--src-dset-path './data/semi_processed' \
--labeled_num 500 \
--num_classes 7 \
--weight-afm 1.0 \
--weight-u 0 \
--suffix 'sev_cates_weight'

run_check 10_0_sev_cate
