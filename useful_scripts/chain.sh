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
:<<comment
python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/train.py \
--src-dset-path './data/semi_processed' \
--labeled_num 500 \
--net "resnet50" \
--threshold 0 \
--num_classes 7 \
--suffix '_t_0'

run_check 500_t_0
sleep 5s

python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/train.py \
--src-dset-path './data/semi_processed' \
--labeled_num 1000 \
--net "resnet50" \
--threshold 0 \
--num_classes 7 \
--suffix '_t_0'

run_check 1000_t_0
sleep 5s

python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/train.py \
--src-dset-path './data/semi_processed' \
--labeled_num 1500 \
--threshold 0 \
--net "resnet50" \
--num_classes 7 \
--suffix '_t_0'

run_check 1500_t_0
sleep 5s

python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/train.py \
--src-dset-path './data/semi_processed' \
--labeled_num 2000 \
--net "resnet50" \
--threshold 0 \
--num_classes 7 \
--suffix '_t_0'

run_check 2500_t_0
sleep 5s
comment

python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/train.py \
--src-dset-path './data/semi_processed' \
--labeled_num 1500 \
--net "ran" \
--num_classes 7 \
--threshold 0 \
--suffix 'tm'

run_check 2500_senet
