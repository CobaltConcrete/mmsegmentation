#!/bin/bash

REQUIRED_MEM_MB=5000     # Change this as needed
GPU_ID=0                 # GPU ID to monitor
VIDEO_LIST="/home/r13qingrong/Projects/URECA/mmsegmentation/demo/VSPWvideofiles_filtered.txt"

wait_for_gpu() {
    echo "Waiting for $REQUIRED_MEM_MB MB of GPU $GPU_ID memory..."
    while true; do
        USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$((GPU_ID+1))p")
        TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sed -n "$((GPU_ID+1))p")
        FREE=$((TOTAL - USED))

        if [ "$FREE" -ge "$REQUIRED_MEM_MB" ]; then
            echo "Sufficient memory available! Continuing..."
            break
        fi

        sleep 10
    done
}

# Run fusion script for each video ID in the file
while IFS= read -r video_id; do
    wait_for_gpu
    echo "Running fusionVSPWargs.py with video ID: $video_id ..."
    python fusionVSPWargs.py "$video_id"
    echo "Finished: $video_id"
done < "$VIDEO_LIST"
