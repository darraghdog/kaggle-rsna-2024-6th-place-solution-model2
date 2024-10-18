#!/bin/bash
# Check if proper arguments are passed
if [ "$#" -ne 2 ]; then
    echo "Usage: infer_model1.sh cfg1 cfg2"
    exit 1
fi

# Assign command line arguments to variables
cfg1=$1
cfg2=$2

for i in {0..3}
do
    # Getting the latest file in the folder
    latest_file=`ls -t /mount/data/models/${cfg2}/fold${i} | head -1`

    if [ -n "$latest_file" ]; then
        cmd="python train.py -C $cfg1 --pretrained_weights /mount/data/models/${cfg2}/fold${i}/$latest_file --pretrained_config $cfg2"
        echo $cmd
        # Uncomment the next line to actually run the command
        # eval $cmd
    else
        echo "No file found in /mount/data/models/${cfg2}/fold${i}"
    fi
done
