current_date_time=$(date +%Y%m%d%H%M%S)
echo "start_time: ""$current_date_time" >> run.log
cp ./object/train.py .

nohup python -u ./train.py \
--src-dset-path './data/semi_processed_two_cates' \
--labeled_num 2500 \
--num_classes 2 > run.log&
