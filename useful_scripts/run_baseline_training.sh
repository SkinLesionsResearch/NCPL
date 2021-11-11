current_date_time=$(date +%Y%m%d%H%M%S)
touch run.log
echo "start_time: ""$current_date_time" >> run.log
# cp ./object/train.py .

nohup python -u ./object/train.py \
--src-dset-path './data/semi_processed' \
--net 'vgg19' \
--labeled_num 2000 \
--threshold 0 \
--suffix "_t_0" \
--num_classes 7 > run.log&
