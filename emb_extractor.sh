cd ./extractor
config_yaml='../model_configs/resnet50_IMAGENET1K_V1.yaml'
model='resnet50'
target_layer_name_visual='fc'
target_layer_name_text='layer4'
weights='IMAGENET1K_V1'
data_path='../concept_library/visual/segments'

python visual_element_emb_extractor.py --config_yaml ${config_yaml} --model ${model} --target_layer_name ${target_layer_name} --weights ${weights} --data-path ${data_path}
python textual_emb_extractor.py --dataset imagenet --concept_set ../concept_library/textual/imagenet_filtered.txt --backbone $model --activation_dir ./embs/${model}/ --feature_layer ${target_layer_name_text}
