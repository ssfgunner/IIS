cd ./IP_training
model_name='resnet50'
target_dataset='imagenet'
target_layer_name='layer4'

# Prototype:
python IP_Prototype_training.py --model_name ${model_name} --in_channels 2048 --concept_path ../concept_extractor/embs/${model_name}/visual_element_emb.pkl \
    --num_concepts 200 --num_classes 1000 --n_epoch 30 --train_pt ../concept_extractor/embs/${model_name}/${target_dataset}_train_${model_name}_${target_layer_name}.pt --val_pt ../concept_extractor/embs/${model_name}/${target_dataset}_val_${model_name}_${target_layer_name}.pt \
    --save_dir ./IP_Prototype/${model_name}

# Textual:
python IP_Text_training.py --dataset ${target_dataset} --backbone ${model_name} --concept_set '../concept_library/textual/${target_dataset}_filtered.txt' --feature_layer ${target_layer_name} --sparsity_ratio --save_dir ./output --activation_dir '../concept_extractor/embs/${model_name}'