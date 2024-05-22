#!/bin/bash

# set the following values
prune_heuristic="sum" # thre or sum

num_layers=4

tap_arch=''
if [ "$num_layers" -eq 12 ]; then
    tap_arch=tap12
elif [ "$num_layers" -eq 4 ]; then
    tap_arch=tap
fi
echo "Arch"
echo $tap_arch

# General config
dataset=m4c_stvqa
model_arch=m4c_lfpr_calc_score
code_dir=/home/cybertron/lfpr/TAP
config=$code_dir/configs/vqa/m4c_stvqa_lfpr/"$tap_arch"/calculate_score.yml 
output_dir=/media/cybertron/fa54fcb6-b5e1-492e-978a-6389519c168a/mmf_save/test
data_dir=/media/cybertron/fa54fcb6-b5e1-492e-978a-6389519c168a/tap_features
org_model=/media/cybertron/fa54fcb6-b5e1-492e-978a-6389519c168a/tap_pt_model/save/finetuned/stvqa_tap_base_best.ckpt

# Pruning config
prune_code_dir=/home/cybertron/lfpr


# retrain config
num_gpu=2
pruned_model_arch=m4c_lfpr_config


python $code_dir/tools/run.py  --tasks vqa --datasets $dataset --model $model_arch \
--config $config \
--run_type train \
--save_dir $output_dir/score \
--resume_file $org_model \
model_attributes."$model_arch".model_data_dir $data_dir \
dataset_attributes."$dataset".data_root_dir $data_dir \
training_parameters.trainer base_trainer_lfpr_calc_score \
training_parameters.score_dir $output_dir/score \
training_parameters.seed 0 \
training_parameters.determine_tmax_q2 true

filename=$output_dir/score/tmax_q2.txt
timestep=$(<"$filename")
echo "selected time step (tmax_q2) is $timestep" 

# Convert content to an integer
timestep=$(echo "$timestep" | tr -d '[:space:]')


#Calculate score

python $code_dir/tools/run.py  --tasks vqa --datasets $dataset --model $model_arch \
--config $config \
--run_type train \
--save_dir $output_dir/score \
--resume_file $org_model \
model_attributes."$model_arch".model_data_dir $data_dir \
dataset_attributes."$dataset".data_root_dir $data_dir \
training_parameters.trainer base_trainer_lfpr_calc_score \
training_parameters.score_dir $output_dir/score \
training_parameters.seed 0  \
model_attributes."$model_arch".selected_time_step $timestep 


                                                                 
if [ "$prune_heuristic" = "sum" ]; then
    echo "" > $output_dir/retrain_lfpr.sh     
    
    echo "Pruning by sum"
    constraints=(0.3 0.4 0.5 0.6 0.7 0.8 0.9)
    retrain_set=(100 90 80 70 60 10 10)
    trainable_layers=(3 3 2 2 1 0 0)
    for i in "${!constraints[@]}"; 
    do
        echo "LFPR pruning with constraint=${constraints[i]}"
        echo "Retraining set size=${retrain_set[i]}%"

        python $prune_code_dir/src/tap_pruning/pruning.py \
        --prune_constraint ${constraints[i]} \
        --output_dir $output_dir/${constraints[i]} \
        --score_dir $output_dir/score \
        --model $org_model \
        --num_layers $num_layers \
        --heuristic $prune_heuristic

        # Check if the file exists


        cfg_file=$output_dir/${constraints[i]}/cfg.txt
        if [ ! -f "$cfg_file" ]; then
            echo "File '$cfg_file' not found."
            exit 1
        fi

        IFS=$'\n'      
        file_content=()
        while IFS= read -r line; do
            file_content+=("$line")
        done < "$cfg_file"
        echo "CFG Head "
        echo ${file_content[0]}
        echo "CFG FFN"
        echo ${file_content[1]}

        echo "retrain"

        retrain_config=$code_dir/configs/vqa/m4c_stvqa_lfpr/"$tap_arch"/retrain_mask_${retrain_set[i]}pct.yml
        
        # Retrain
        echo -e """python -m torch.distributed.launch --nproc_per_node $num_gpu $code_dir/tools/run.py --tasks vqa --datasets $dataset --model $pruned_model_arch \\
--config $retrain_config \\
--save_dir $output_dir/${constraints[i]}/retrain  \\
--resume_file $output_dir/${constraints[i]}/pruned_model.ckpt \\
training_parameters.distributed True \\
training_parameters.trainer base_trainer_ReSt_retrain \\
training_parameters.layers ${trainable_layers[i]} \\
dataset_attributes."$dataset".data_root_dir $data_dir \\
model_attributes."$pruned_model_arch".model_data_dir $data_dir \\
model_attributes."$pruned_model_arch".mmt.cfg_head ${file_content[0]} \\
model_attributes."$pruned_model_arch".mmt.cfg_ffn ${file_content[1]} \\
model_attributes."$pruned_model_arch".mmt.mask True""" >> $output_dir/retrain_lfpr.sh     

        echo -e "" >> $output_dir/retrain_lfpr.sh     
    done
    bash $output_dir/retrain_lfpr.sh     


elif [ "$prune_heuristic" = "thre" ]; then
    echo "" > $output_dir/retrain_thre.sh   
    echo "Pruning by threshold"
    percentiles=(10 30 50 70 90)
    retrain_set=(10 10 10 73 100)
    trainable_layers=(0 0 0 2 3)
    for i in "${!percentiles[@]}"; 
    do
        echo "LFPR(thre) pruning with constraint=${percentiles[i]}"
        echo "Retraining set size=${retrain_set[i]}%"


        python $prune_code_dir/src/tap_pruning/pruning.py \
        --percentile ${percentiles[i]} \
        --output_dir $output_dir/${percentiles[i]} \
        --score_dir $output_dir/score \
        --model $org_model \
        --num_layers $num_layers \
        --heuristic $prune_heuristic


        # Check if the file exists


        cfg_file=$output_dir/${percentiles[i]}/cfg.txt
        if [ ! -f "$cfg_file" ]; then
            echo "File '$cfg_file' not found."
            exit 1
        fi

        IFS=$'\n'      
        file_content=()
        while IFS= read -r line; do
            file_content+=("$line")
        done < "$cfg_file"
        echo "CFG Head "
        echo ${file_content[0]}
        echo "CFG FFN"
        echo ${file_content[1]}

        echo "retrain"

        retrain_config=$code_dir/configs/vqa/m4c_stvqa_lfpr/"$tap_arch"/retrain_mask_${retrain_set[i]}pct.yml
        
        # Write the retraining
        echo -e """python -m torch.distributed.launch --nproc_per_node $num_gpu $code_dir/tools/run.py --tasks vqa --datasets $dataset --model $pruned_model_arch \\
--config $retrain_config \\
--save_dir $output_dir/${percentiles[i]}/retrain  \\
--resume_file $output_dir/${percentiles[i]}/pruned_model.ckpt \\
training_parameters.distributed True \\
training_parameters.trainer base_trainer_ReSt_retrain \\
training_parameters.layers ${trainable_layers[i]} \\
dataset_attributes."$dataset".data_root_dir $data_dir \\
model_attributes."$pruned_model_arch".model_data_dir $data_dir \\
model_attributes."$pruned_model_arch".mmt.cfg_head ${file_content[0]} \\
model_attributes."$pruned_model_arch".mmt.cfg_ffn ${file_content[1]} \\
model_attributes."$pruned_model_arch".mmt.mask True""" >> $output_dir/retrain_thre.sh   

        echo -e "" >> $output_dir/retrain_thre.sh   

    done
    bash $output_dir/retrain_thre.sh   

fi


