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

