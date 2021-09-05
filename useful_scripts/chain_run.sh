if [ $# -eq 0 ]
	then echo "No arguments supplied, please chain script to run the chain jobs"
	exit
fi
current_date_time=$(date +%Y%m%d%H%M%S)
touch chain.log
echo "start_time: ""$current_date_time" >> chain.log
cp ./object/train.py .
nohup ./$1 > chain.log &
