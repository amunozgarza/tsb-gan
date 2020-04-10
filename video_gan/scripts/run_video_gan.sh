#!/bin/bash
unset CUDA_VISIBLE_DEVICES
gpuName="$(nvidia-smi --query-gpu="gpu_name" --format=csv,noheader,nounits -i 0)"
echo "$(date) - Training started on host ${HOSTNAME} on an ${gpuName}"


##################################################################### 
# Train on local machine
if [ "$1" != "local" ] && [ "$2" != "local" ] && [ "$3" != "local" ]; then
    cd $PBS_O_WORKDIR
fi


##################################################################### 
save_p="path/to/folder/"
log_path="${save_p}/logs/ucf101_strided/"
save_weights="${save_p}/models/ucf101_strided/"
save_samples='samples/ucf101_strided/'
dataset_path="${save_p}/ucf101_train.lst"

python -W ignore::UserWarning ${save_p}/train.py --batch_size 48  --dataset ucf101 --adv_loss hinge --save_path ${save_p} --d_steps_per_iter 2 --g_steps_per_iter 1 --batch_size_in_gpu 6 --model_weights_dir ${save_weights} --sample_images_dir ${save_samples} --d_lr 2e-4 --g_lr 5e-5 --g_conv_dim 96 --d_conv_dim 96 --model_save_step 2000 --data_path ${dataset_path} --pretrained_path "${save_p}/ckpt_0020700.pth" --pretrained True --log_path ${log_path} --save_n_images 6 --attention True --freeze False --z_dim 120

