cd ./extractor
python segment_extractor.py --data_root $1 --save_root ../concept_library/visual/segments
python patch_extractor.py --data_root $1 --save_root ../concept_library/visual/patches
