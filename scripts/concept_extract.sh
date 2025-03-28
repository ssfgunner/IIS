# Taking the validation set for concept extraction:
cd ./concept_extractor
path_of_concept_dataset="../datasets/imagenet/val"
# extract image segments
python segment_extractor.py --data_root ${path_of_concept_dataset} --save_root ../concept_library/prototype/segments
# or extract image patches
python patch_extractor.py --data_root ${path_of_concept_dataset} --save_root ../concept_library/prototype/patches