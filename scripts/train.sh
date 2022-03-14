GPU=0

function run_reduced_svhn {
    DATASET=reduced_svhn
    MODEL=wresnet28_10
    EPOCH=180
    BATCH=128
    LR=0.05
    WD=0.01
    CUTOUT=0
    TRAIN_PORTION=1
    GF=<PATH_TO_G_MODEL_DIR>
    H=<PATH_TO_H_MODEL_DIR>
    SDN=reduced_svhn
    GFN=wresnet40_2
    DELTA=0.4
    TEMP=2
    KOPS=2
}

function run_reduced_cifar10 {
    DATASET=reduced_cifar10
    MODEL=wresnet28_10
    EPOCH=240
    BATCH=128
    LR=0.1
    WD=0.0005
    CUTOUT=16
    TRAIN_PORTION=1
    GF=<PATH_TO_G_MODEL_DIR>
    H=<PATH_TO_H_MODEL_DIR>
    SDN=reduced_cifar10
    GFN=wresnet40_2
    DELTA=0.3
    TEMP=3
    KOPS=2
}

if [ $1 = "reduced_cifar10" ]; then
    run_reduced_cifar10
elif [ $1 = "reduced_svhn" ]; then
    run_reduced_svhn
fi

SAVE=${DATASET}_${MODEL}_${BATCH}_${EPOCH}_cutout_${CUTOUT}_lr${LR}_wd${WD}_kops_${KOPS}_TEMP_${TEMP}_${DELTA}
python ada_aug/train.py --temperature ${TEMP} --delta ${DELTA} --search_dataset ${SDN} --gf_model_path ${GF} --h_model_path ${H} --gf_model_name ${GFN} --k_ops ${KOPS} --report_freq 10 --num_workers 8 --epochs ${EPOCH} --batch_size ${BATCH} --learning_rate ${LR} --dataset ${DATASET} --model_name ${MODEL} --save ${SAVE} --gpu ${GPU} --weight_decay ${WD} --train_portion 1 --use_parallel
