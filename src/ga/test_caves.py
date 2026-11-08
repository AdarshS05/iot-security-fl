import sys
from pathlib import Path
import random

sys.path.insert(0, '/data/iot-malware-fl/iot-malware-abhi/src/ga')
sys.path.insert(0, '/data/iot-malware-fl/iot-malware-abhi/src/assembly2')

from genetic_algorithm_mutator import find_code_cave, find_trampoline_insertion_points, _j_type_reachable, generate_junk_instruction_stream

bin_path = Path('/data/iot-malware-fl/iot-malware-abhi/data/raw/raw_mips/1a19777cc215e563f04f96d2bd8095fed8a3f04167426beaa61968ce682580eb')

cave = find_code_cave(bin_path, min_size=16)
seg = cave['segment']
aligned_file_offset = (cave['file_offset'] + 3) & ~3
alignment_padding = aligned_file_offset - cave['file_offset']
usable = cave['usable'] - alignment_padding
trampoline_vaddr = seg['p_vaddr'] + (aligned_file_offset - seg['p_offset'])
print('Aligned file offset:', hex(aligned_file_offset), 'alignment_padding:', alignment_padding, 'usable:', usable)
print('Aligned trampoline_vaddr:', hex(trampoline_vaddr), 'mod 4:', trampoline_vaddr % 4)

pairs = find_trampoline_insertion_points(bin_path)

success_count = 0
for ins_a, ins_b in pairs:
    return_vaddr = ins_b.address + 4
    n_junk = 4
    junk_bytes = generate_junk_instruction_stream(random.Random(42), n_junk)
    return_jump_vaddr = trampoline_vaddr + len(junk_bytes)
    reach1 = _j_type_reachable(return_jump_vaddr + 4, return_vaddr)
    reach2 = _j_type_reachable(ins_a.address + 4, trampoline_vaddr)
    if reach1 and reach2:
        success_count += 1

print(f'With 4-byte alignment, valid reachable pairs: {success_count} / {len(pairs)}')
