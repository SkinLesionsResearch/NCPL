current_date_time=$(date +%Y%m%d%H%M%S)
touch testing.log
echo "start_time: ""$current_date_time" >> testing.log
# cp ./object/test.py .

nohup python -u ./object/test.py \
--batch_size 32 \
--which 'all' \
--img_dir './../ups/data_ham10000/datasets/' \
--save_dir 'testing_output' \
--num_classes 7 > testing.log&
