# USAGE bash train_parallel.sh "0 1 2 3" "0 1 2 5" -C cfg_isic24 
# 

set -e
ARRAY1=($1) ; shift
ARRAY2=($1) ; shift

for i in "${!ARRAY1[@]}"
do
    FOLD=${ARRAY1[i]}
    GPU=${ARRAY2[i]}
    CUDA_VISIBLE_DEVICES=$GPU python train.py --fold $FOLD "$@" & 
    
done
wait
