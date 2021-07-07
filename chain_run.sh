current_date_time=$(date +%Y%m%d%H%M%S)
touch chain.log
echo "start_time: ""$current_date_time" >> chain.log
cp ./object/train.py .
nohup ./chain.sh > chain.log &
