if [ $# -eq 0 ]
	then echo "No arguments supplied, please input path to tb"
	exit
fi
nohup tensorboard --logdir=$1 > tb.log&
