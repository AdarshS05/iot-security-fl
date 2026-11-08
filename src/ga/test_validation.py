import sys
from pathlib import Path
import random
import shutil

sys.path.insert(0, '/data/iot-malware-fl/iot-malware-abhi/src/ga')
sys.path.insert(0, '/data/iot-malware-fl/iot-malware-abhi/src/assembly2')

from genetic_algorithm_mutator import (
    find_code_cave,
    find_trampoline_insertion_points,
    apply_code_cave_patch,
    apply_trampoline_patch,
    Validator,
    generate_junk_instruction_stream,
)
import lief

bin_path = Path('/data/iot-malware-fl/iot-malware-abhi/data/raw/raw_mips/1a19777cc215e563f04f96d2bd8095fed8a3f04167426beaa61968ce682580eb')
work_path = Path('/tmp/test_variant.elf')
shutil.copyfile(bin_path, work_path)

probe = lief.parse(str(bin_path))

cave = find_code_cave(bin_path, min_size=16)
seg = cave['segment']
aligned_file_offset = (cave['file_offset'] + 3) & ~3
grow = 32
junk_bytes = generate_junk_instruction_stream(random.Random(42), 8)

params = {
    'file_offset': aligned_file_offset,
    'new_bytes': junk_bytes.hex(),
    'segment_index': seg['index'],
    'endian': cave['endian'],
    'e_phoff': cave['e_phoff'],
    'e_phentsize': cave['e_phentsize'],
    'original_filesz': seg['p_filesz'],
    'original_memsz': seg['p_memsz'],
    'segment_vaddr': seg['p_vaddr'],
}

mods = apply_code_cave_patch(work_path, params)

# Append overlay padding
with open(work_path, 'ab') as f:
    f.write(b'\x90' * 128)

validator = Validator(
    expected_machine=probe.header.machine_type,
    baseline_path=str(bin_path),
    allow_byte_modifications=True
)

allowed_seg_growth = {seg['p_vaddr']: grow}
val_res = validator.validate(
    str(work_path),
    allowed_modifications=mods,
    allowed_segment_growth=allowed_seg_growth
)
print('With Overlay + Code Cave: valid =', val_res.valid, 'reason =', val_res.reason)
if not val_res.valid:
    print('Failed checks / reason:', val_res.reason)
    print('Instruction changes check:', val_res.metrics.get('instruction_changes'))
