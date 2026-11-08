import sys
from pathlib import Path
import numpy as np
import shutil
import random

sys.path.insert(0, '/data/iot-malware-fl/iot-malware-abhi/src/ga')
sys.path.insert(0, '/data/iot-malware-fl/iot-malware-abhi/src/assembly2')

from genetic_algorithm_mutator import (
    SiameseSVMScorer,
    find_code_cave,
    apply_code_cave_patch
)

model_path = '/data/iot-malware-fl/iot-malware-abhi/models/siamese_svm_n1/siamese_svm_fewshot.joblib'
matrix_path = '/data/iot-malware-fl/iot-malware-abhi/data/features/tfidf_n1/X_train.npz'
labels_path = '/data/iot-malware-fl/iot-malware-abhi/data/features/tfidf_n1/train_samples.csv'

scorer = SiameseSVMScorer(
    model_path=model_path,
    matrix_path=matrix_path,
    labels_path=labels_path,
    random_state=42
)

bin_path = Path('/data/iot-malware-fl/iot-malware-abhi/data/raw/raw_mips/1a19777cc215e563f04f96d2bd8095fed8a3f04167426beaa61968ce682580eb')
base_det = scorer.predict_details(str(bin_path))
print(f"Base: margin={base_det['score']:.4f} pred={base_det['prediction']}")

# Benign instructions:
# MIPS instructions common in benign firmware:
# lui $v0, 0x0040 (0x3c020040)
# addiu $v0, $v0, 0x1000 (0x24421000)
# move $a0, $v0 (0x00022021)
# nop (0x00000000)
# addu $t0, $t1, $t2
benign_words = [
    0x3c0f0041, # lui $t7, 0x0041 (matches 'lui t7' and 't7 0x41'!)
    0x3c190040, # lui $t9, 0x0040
    0x27180020, # addiu $t8, $t8, 32
    0x00000000, # nop
    0x00001021, # move $v0, $zero
    0x24040001, # li $a0, 1
    0x24050002, # li $a1, 2
    0x00851020, # add $v0, $a0, $a1
]

def make_benign_stream(count):
    res = bytearray()
    for i in range(count):
        w = benign_words[i % len(benign_words)]
        res.extend(w.to_bytes(4, 'little'))
    return bytes(res)

for n_inst in [16, 32, 64]:
    work_path = Path(f'/tmp/test_benign_cave_{n_inst}.elf')
    shutil.copyfile(bin_path, work_path)
    cave = find_code_cave(bin_path, min_size=n_inst * 4)
    seg = cave['segment']
    aligned_offset = (cave['file_offset'] + 3) & ~3
    b_stream = make_benign_stream(n_inst)
    params = {
        'file_offset': aligned_offset,
        'new_bytes': b_stream.hex(),
        'segment_index': seg['index'],
        'endian': cave['endian'],
        'e_phoff': cave['e_phoff'],
        'e_phentsize': cave['e_phentsize'],
        'original_filesz': seg['p_filesz'],
        'original_memsz': seg['p_memsz'],
        'segment_vaddr': seg['p_vaddr'],
    }
    apply_code_cave_patch(work_path, params)
    det = scorer.predict_details(str(work_path))
    delta = base_det['score'] - det['score']
    print(f"Benign Cave {n_inst} inst: margin={det['score']:.4f} pred={det['prediction']} delta={delta:.4f}")
