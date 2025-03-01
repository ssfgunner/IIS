# [ICLR 2025 Spotlight] Enhancing Pre-trained Representation Classifiability can Boost its Interpretability

 ![framework](./framework.png)

## Introduction

The visual representation of a pre-trained model prioritizes the classifiability on downstream tasks. However, widespread applications for pre-trained visual models have proposed new requirements for representation interpretability. It remains unclear whether the pre-trained representations can achieve high interpretability and classifiability simultaneously. To answer this question, we quantify the representation interpretability by leveraging its correlation with the ratio of interpretable semantics within representations. Given the pre-trained representations, only the interpretable semantics can be captured by interpretations, whereas the uninterpretable part leads to information loss. Based on this fact, we propose the Inherent Interpretability Score (IIS) that evaluates the information loss, measures the ratio of interpretable semantics, and quantifies the representation interpretability. In the evaluation of the representation interpretability with different classifiability, we surprisingly discover that the interpretability and classifiability are positively correlated, i.e., representations with higher classifiability provide more interpretable semantics that can be captured in the interpretations. This observation further supports two benefits to the pre-trained representations. First, the classifiability of representations can be further improved by fine-tuning with interpretability maximization. Second, with the classifiability improvement for the representations, we obtain predictions based on their interpretations with less accuracy degradation. The discovered positive correlation and corresponding applications show that practitioners can unify the improvements in interpretability and classifiability for pre-trained vision models.

## Prerequisites

Main packages are:

- PyYAML 6.0
- scikit-image 0.20.0
- scikit-learn 1.3.0
- scipy 1.10.1
- timm 0.6.13
- torch 1.12.1
- torchaudio 0.12.1
- torchvision 0.13.1

## Usage

We provide the code for computing IIS of representations pre-trained on ImageNet in this repository.

We present the codes for Prototype and Text concept libraries, codes for other concept libraries will be provided later.

Taking representations from ResNet-50 for example, we first extract visual elements (segments, patches) from images:

```bash
cd ./extractor
python segment_extractor.py --data_root ${dataset_traindir} --save_root ../concept_library/visual/segments
python patch_extractor.py --data_root ${dataset_traindir} --save_root ../concept_library/visual/patches
```

Then, the features of concept textual and visual concepts are obtained by:

```bash
cd ./extractor
config_yaml='../model_configs/resnet50_IMAGENET1K_V1.yaml'
model='resnet50'
target_layer_name_visual='fc'
target_layer_name_text='layer4'
weights='IMAGENET1K_V1'
data_path='../concept_library/visual/segments'

python visual_element_emb_extractor.py --config_yaml ${config_yaml} --model ${model} --target_layer_name ${target_layer_name} --weights ${weights} --data-path ${data_path}
python textual_emb_extractor.py --dataset imagenet --concept_set ../concept_library/textual/imagenet_filtered.txt --backbone $model --activation_dir ./embs/${model}/ --feature_layer ${target_layer_name_text}
```

Finally, we evaluate the prediction accuracy of interpretations with different sparsity ratio and compute the IIS of ResNet-50 (taking prototype concept library for example):

```bash
cd ./IP_training
model_name=resnet50
python IP_Prototype_training.py --model_name ${model_name} --in_channels 2048 --concept_path ../extractor/embs/${model_name}/visual_element_emb.pkl \
    --num_concepts 200 --num_classes 1000 --n_epoch 30 --train_pt ../extractor/embs/${model_name}/imagenet_train_resnet50_layer4.pt --val_pt ../extractor/embs/${model_name}/imagenet_val_resnet50_layer4.pt \
    --save_dir ./IP_Prototype/${model_name}
    
cd ../
python iis_computing.py --model_root ./IP_Prototype/${model_name}
```

For ease of running, we provide the following scripts:

```bash
bash concept_extract.sh
bash emb_extractor.sh
bash ip_train.sh
```
