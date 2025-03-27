import os
import json

with open('./ucf101_concepts.json', 'r') as f:
    cls2concepts = json.load(f)
print(cls2concepts.keys())

concept_list = []

for cls_name in cls2concepts:
    concept_list += cls2concepts[cls_name][:50]
concept_list = [item + '\n' for item in concept_list]
with open('./ucf101_filtered.txt', 'w') as f:
    f.writelines(concept_list)