#!/bin/bash

# File used to record experimental data
record_file="ExperimentRecord.txt"
if [ -f $record_file ]; then
  rm $record_file
fi

# The program executes multiple times, averages the results, and formats them.
for iteration in `seq 4`;
do
     python exe_main.py \
        --device cuda:0 >> $record_file
done


cur_file_path=$(pwd)
python output.py --record_root_path $cur_file_path --record_file_name $record_file



