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