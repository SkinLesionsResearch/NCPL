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
--labeled_num 1500 \
--num_classes 7 \
--net resnet18 \
--suffix 'test_nets'

run_check test_nets_1
sleep 5s


python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/train.py \
--src-dset-path './data/semi_processed' \
--labeled_num 1500 \
--num_classes 7 \
--net resnet34 \
--suffix 'test_nets'

run_check test_nets_2
sleep 5s


python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/train.py \
--src-dset-path './data/semi_processed' \
--labeled_num 2000 \
--num_classes 7 \
--net resnet18 \
--suffix 'test_nets'

run_check test_nets_3
sleep 5s


python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/train.py \
--src-dset-path './data/semi_processed' \
--labeled_num 2000 \
--num_classes 7 \
--net resnet34 \
--suffix 'test_nets'

run_check test_nets_4
sleep 5s


python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/train.py \
--src-dset-path './data/semi_processed' \
--labeled_num 2500 \
--num_classes 7 \
--net resnet18 \
--suffix 'test_nets'

run_check test_nets_5
sleep 5s


python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/train.py \
--src-dset-path './data/semi_processed' \
--labeled_num 2500 \
--num_classes 7 \
--net resnet34 \
--suffix 'test_nets'

run_check test_nets_6
sleep 5s
