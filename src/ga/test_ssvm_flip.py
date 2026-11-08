import sys
from pathlib import Path
import numpy as np
import shutil

sys.path.insert(0, '/data/iot-malware-fl/iot-malware-abhi/src/ga')
sys.path.insert(0, '/data/iot-malware-fl/iot-malware-abhi/src/assembly2')

from genetic_algorithm_mutator import (
    SiameseSVMScorer,
    find_code_cave,
    generate_junk_instruction_stream,
    apply_code_cave_patch
)
import random

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
base_details = scorer.predict_details(str(bin_path))
print(f"Base margin={base_details['score']:.4f} pred={base_details['prediction']}")

# Let's see what happens if we patch the cave with different numbers of instructions
for n_inst in [8, 16, 32, 64]:
    work_path = Path(f'/tmp/test_cave_{n_inst}.elf')
    shutil.copyfile(bin_path, work_path)
    cave = find_code_cave(bin_path, min_size=n_inst * 4)
    seg = cave['segment']
    aligned_offset = (cave['file_offset'] + 3) & ~3
    align_pad = aligned_offset - cave['file_offset']
    junk = generate_junk_instruction_stream(random.Random(42), n_inst)
    params = {
        'file_offset': aligned_offset,
        'new_bytes': junk.hex(),
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
    print(f"Cave {n_inst} inst: margin={det['score']:.4f} pred={det['prediction']} delta={base_details['score'] - det['score']:.4f}")
