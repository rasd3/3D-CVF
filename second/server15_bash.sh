_model_name="test"
_input_config="first_stage.fhd.onestage.config"
_use_fusion=True
_use_second=False

LOOP_O="1"
for c in $LOOP_O
do
    CUDA_VISIBLE_DEVICES=$1 python pytorch/train.py train\
    --config_path="configs/${_input_config}"\
    --model_dir="logs/${_model_name}_gpu$1_$c"\
    --use_fusion=${_use_fusion}\
    --use_second=${_use_second}
done


# CUDA_VISIBLE_DEVICES=1 python pytorch/train.py evaluate --config_path="configs/twostage_iou/demo_server_2st_fusion_test.fhd.onestage.config" --model_dir="logs/end-to-end-mod_lr2_gpu1_2" --ckpt_path="logs/end-to-end-mod_lr2_gpu1_2/voxelnet-4024.tckpt"   --use_fusion True --use_endtoend=True --predict_test=True
# CUDA_VISIBLE_DEVICES=1 python pytorch/train.py evaluate --config_path="configs/twostage_iou/demo_server_2st_fusion_test.fhd.onestage.config" --model_dir="logs/end-to-end-mod_lr2_gpu1_2" --ckpt_path="logs/end-to-end-mod_lr2_gpu1_2/voxelnet-4024.tckpt"   --use_fusion True --use_endtoend=True 