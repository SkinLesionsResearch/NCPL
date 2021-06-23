current_date_time=`date +%Y%m%d%H%M%S`
echo "start_time: "$current_date_time >> run.log
cp ./object/train.py .
nohup python -u ./train.py --labeled_num 500 > run.log&
