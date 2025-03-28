cd ./concept_extractor
# for Prototype concepts
config_yaml='../model_configs/resnet50_IMAGENET1K_V1.yaml'
model='resnet50'
target_layer_name='layer4'
weights='IMAGENET1K_V1'
target_dataset='imagenet'
prototype_concept_path='../concept_library/prototype/segments'
python visual_element_emb_extractor.py --config_yaml ${config_yaml} --model ${model} --target_layer_name ${target_layer_name} --weights ${weights} --data-path ${prototype_concept_path}

# for Text concepts
text_concept_path='../concept_library/textual/imagenet_filtered.txt'
python textual_emb_extractor.py --dataset ${target_dataset} --concept_set ${text_concept_path} --backbone ${model} --activation_dir ./embs/${model}/ --feature_layer ${target_layer_name}