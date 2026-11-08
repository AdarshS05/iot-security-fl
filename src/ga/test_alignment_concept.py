import sys
from pathlib import Path
import random

sys.path.insert(0, '/data/iot-malware-fl/iot-malware-abhi/src/ga')
sys.path.insert(0, '/data/iot-malware-fl/iot-malware-abhi/src/assembly2')

from genetic_algorithm_mutator import find_code_cave, find_trampoline_insertion_points, _j_type_reachable

bin_path = Path('/data/iot-malware-fl/iot-malware-abhi/data/raw/raw_mips/1a19777cc215e563f04f96d2bd8095fed8a3f04167426beaa61968ce682580eb')

cave = find_code_cave(bin_path, min_size=16)
seg = cave['segment']
seg_file_end = seg['p_offset'] + seg['p_filesz']
align_pad = (4 - (seg_file_end % 4)) % 4
trampoline_vaddr = seg['p_vaddr'] + seg['p_filesz'] + align_pad

print(f"seg_file_end={hex(seg_file_end)} align_pad={align_pad} trampoline_vaddr={hex(trampoline_vaddr)} mod4={trampoline_vaddr % 4}")

pairs = find_trampoline_insertion_points(bin_path)
reach_count = 0
for ins_a, ins_b in pairs:
    return_vaddr = ins_b.address + 4
    return_jump_vaddr = trampoline_vaddr + 32 # say 8 words
    if _j_type_reachable(ins_a.address + 4, trampoline_vaddr) and _j_type_reachable(return_jump_vaddr + 4, return_vaddr):
        reach_count += 1

print(f"Reachable trampoline pairs: {reach_count} / {len(pairs)}")
