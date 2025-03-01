cd ./IP_training
model_name=resnet50
python IP_Prototype_training.py --model_name ${model_name} --in_channels 2048 --concept_path ../extractor/embs/${model_name}/visual_element_emb.pkl \
    --num_concepts 200 --num_classes 1000 --n_epoch 30 --train_pt ../extractor/embs/${model_name}/imagenet_train_resnet50_layer4.pt --val_pt ../extractor/embs/${model_name}/imagenet_val_resnet50_layer4.pt \
    --save_dir ./IP_Prototype/${model_name}

cd ../
python iis_computing.py --model_root ./IP_Prototype/${model_name}