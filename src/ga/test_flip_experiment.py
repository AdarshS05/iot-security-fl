import sys
from pathlib import Path
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

benign_words = [
    0x3c0f0041, # lui $t7, 0x0041
    0x25ef0000, # addiu $t7, $t7, 0
    0x8df80000, # lw $t8, 0($t7)
    0x3c190040, # lui $t9, 0x0040
    0x27180020, # addiu $t8, $t8, 32
    0x00000000, # nop
    0x00001021, # move $v0, $zero
    0x24040001, # li $a0, 1
    0x24050002, # li $a1, 2
    0x00851020, # add $v0, $a0, $a1
    0x0c000000, # jal 0
    0x00000000, # nop
    0x3c100000, # lui $s0, 0
    0x8e10001c, # lw $s0, 0x1c($s0)
    0x00000000, # nop
]

def make_benign_stream(count):
    res = bytearray()
    for i in range(count):
        w = benign_words[i % len(benign_words)]
        res.extend(w.to_bytes(4, 'little'))
    return bytes(res)

for bin_name in [
    '0068070679dc07f54aecc4073bfde92ef9c48438b86eba7c1df979ffb5abca49',
    '002437814326bf60f743a18e5a0ee7cb94214c23326db51ccb56fd43a1241bcd',
    '006e912ac8d276d5c7bc5ec8ec6b216ec89d6eefb4081b3734a946261c472855'
]:
    matches = list(Path('/data/iot-malware-fl/iot-malware-abhi/data/raw/raw_mips').glob(f'{bin_name[:16]}*'))
    if not matches:
        continue
    bin_path = matches[0]
    base_det = scorer.predict_details(str(bin_path))
    cave = find_code_cave(bin_path, min_size=16)
    if not cave:
        print(f"{bin_path.name[:16]}: base={base_det['score']:.4f} NO CAVE")
        continue
    seg = cave['segment']
    aligned_offset = (cave['file_offset'] + 3) & ~3
    max_inst = min(128, cave['usable'] // 4)
    print(f"\n{bin_path.name[:16]}: base={base_det['score']:.4f} usable_inst={max_inst}")

    for n_inst in [32, 64, max_inst]:
        if n_inst <= 0: continue
        work_path = Path(f'/tmp/test_flip_{bin_name[:8]}_{n_inst}.elf')
        shutil.copyfile(bin_path, work_path)
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
        print(f"  Cave {n_inst} inst: margin={det['score']:+.4f} pred={det['prediction']} delta={delta:+.4f}")
        if det['prediction'] == 0:
            print("  >>> SUCCESSFUL LABEL FLIP FOR SSVM! <<<")
