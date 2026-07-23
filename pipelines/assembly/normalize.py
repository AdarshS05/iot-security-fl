import os
import re
from collections import Counter

INPUT_DIR = "data/processed_assembly/benign_mips"
OUTPUT_DIR = "data/normalized_assembly/benign_mips"


REGISTER_MAP = {
    'a0': 'ARG_REG', 'a1': 'ARG_REG', 'a2': 'ARG_REG', 'a3': 'ARG_REG',
    'v0': 'RET_REG', 'v1': 'RET_REG',
    't0': 'SCRATCH_REG', 't1': 'SCRATCH_REG', 't2': 'SCRATCH_REG', 
    't3': 'SCRATCH_REG', 't4': 'SCRATCH_REG', 't5': 'SCRATCH_REG', 
    't6': 'SCRATCH_REG', 't7': 'SCRATCH_REG', 't8': 'SCRATCH_REG', 
    't9': 'SCRATCH_REG', 
    's0': 'SAVED_REG', 's1': 'SAVED_REG', 's2': 'SAVED_REG', 
    's3': 'SAVED_REG', 's4': 'SAVED_REG', 's5': 'SAVED_REG', 
    's6': 'SAVED_REG', 's7': 'SAVED_REG',
    'sp': 'STACK_PTR',
    'ra': 'RETURN_ADDR',
    'gp': 'GLOBAL_PTR',
    'fp': 'FRAME_PTR',
    'zero': 'ZERO_REG',
    'at': 'ASSEM_TEMP',
    'k0': 'KERNEL_REG', 'k1': 'KERNEL_REG',
}


def normalize_immediate(value_str):
    try:
        if value_str is None:
            value_str= "0"
        val = abs(int(value_str,0))
        
        if val == 0:
            return 'ZERO_IMM'
        elif val <= 15:
            return 'SMALL_IMM'
        elif val <= 255:
            return 'MED_IMM'
        elif val > 65535:
            return 'ADDR_IMM'
        else:
            return 'LARGE_IMM'
    except ValueError:
        return 'IMM'
        
def normalize_operand(op_str):
    op_str = op_str.strip()
        
    mem_match = re.match(r"(?:(-?(?:0x[\da-fA-F]+|\d+))?)\(([\$\w]+)\)$",op_str)
    
    if mem_match:
        offset_str, reg_str = mem_match.groups()
        if offset_str is None or offset_str=="":
            offset="0"
        reg_str= reg_str.lstrip("$")
        reg_token = REGISTER_MAP.get(reg_str, 'REG')
        offset_token = normalize_immediate(offset_str)
        return f'MEM_{offset_token}_{reg_token}'
        
    if op in REGISTER_MAP:
        return REGISTER_MAP[op]
        
    if re.fullmatch(r'0x[\da-fA-F]+', op_str):
        return "ADDR_IMM"
        
    if op.startswith("sym.") or op.startswith("fcn."):
        return 'FUNC'
      
    if op.startswith("str."):
        return 'STRING' 
    
    if re.fullmatch(r"-?(?:0x[\da-fA-F]+|\d+)", op):
        return normalize_immediate(op)
        
    return op
    
tot_files=0
tot_instrs=0
vocab_b=Counter()
vocab_a=Counter()

for file in sorted(os.listdir(INPUT_DIR)):
    if not file.endswith(".asm"):
        continue
    tot_files+=1
    
    input_path=os.path.join(INPUT_DIR,file)
    output_path=os.path.join(OUTPUT_DIR,file)
    
    normalized = []
    
    with open(input_path, "r") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            parts=line.split(None,1)
            mnemonic=parts[0].lower()
            vocab_b[mnemonic]+=1
            tokens=[mnemonic]
            if len(parts)>1:
                operands = [x.strip() for x in parts[1].split(",")]
                for op in operands:
                    token=normalize_operand(op)
                    tokens.append(token)
            normalized.append(" ".join(tokens))
            tot_instrs+=1
            for t in tokens:
                vocab_a[t]+=1
    with open(output_path,"w") as f:
        f.write("\n".join(normalized))
        
print("NORMALIZATION SUMMARY")

print(f"Files Processed          : {tot_files}")
print(f"Instructions Processed   : {tot_instrs}")
print(f"Unique Mnemonics         : {len(vocab_b)}")
print(f"Normalized Vocabulary    : {len(vocab_a)}")


