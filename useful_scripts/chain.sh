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
--labeled_num 1000 \
--weight-afm 0.9 \
--weight-u 0.1 \
--num_classes 7 \
--suffix "acfi_1000"

run_check 1000_acfi_0901

python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/object/train.py \
--src-dset-path './data/semi_processed' \
--labeled_num 1000 \
--weight-afm 0.7 \
--weight-u 0.3 \
--num_classes 7 \
--suffix "acfi_1000"
run_check 1000_acfi_0703

python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/object/train.py \
--src-dset-path './data/semi_processed' \
--labeled_num 2500 \
--weight-afm 0.9 \
--weight-u 0.1 \
--num_classes 7 \
--suffix "acfi_2500"

run_check 2500_acfi_0901

python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/object/train.py \
--src-dset-path './data/semi_processed' \
--labeled_num 2500 \
--weight-afm 0.7 \
--weight-u 0.3 \
--num_classes 7 \
--suffix "acfi_2500"

run_check 2500_acfi_0703

python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/object/train.py \
--src-dset-path './data/semi_processed' \
--labeled_num 2500 \
--weight-afm 0.6 \
--weight-u 0.4 \
--num_classes 7 \
--suffix "acfi_2500"

run_check 2500_acfi_0604

python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/object/train.py \
--src-dset-path './data/semi_processed' \
--labeled_num 2500 \
--weight-afm 0.5 \
--weight-u 0.5 \
--num_classes 7 \
--suffix "acfi_2500"

run_check 2500_acfi_0505

python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/object/train.py \
--src-dset-path './data/semi_processed' \
--labeled_num 2500 \
--weight-afm 0.4 \
--weight-u 0.6 \
--num_classes 7 \
--suffix "acfi_2500"

run_check 2500_acfi_0406

python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/object/train.py \
--src-dset-path './data/semi_processed' \
--labeled_num 2500 \
--weight-afm 0.3 \
--weight-u 0.7 \
--num_classes 7 \
--suffix "acfi_2500"

run_check 2500_acfi_0307

python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/object/train.py \
--src-dset-path './data/semi_processed' \
--labeled_num 2500 \
--weight-afm 0.1 \
--weight-u 0.9 \
--num_classes 7 \
--suffix "acfi_2500"

run_check 2500_acfi_0109
