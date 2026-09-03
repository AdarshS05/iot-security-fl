import os
import csv
import statistics

ASSEMBLY_DIR = "data/processed_assembly/benign_mips"
REPORT= "disassembly_report.csv"

MIN_INSTRUCTIONS=50
MIN_UNIQUE_INSTRUCTIONS=5

def validate_file(path):
	result = {
		"filename": os.path.basename(path),
		"status": "VALID",
		"instruction_count": 0, 
		"unique_instructions": 0,
		"size": os.path.getsize(path),
		"reason": ""
	}
	
	if result["size"]==0:
		result["status"]="EMPTY"
		result["reason"]="Zero byte assembly file"
		return result
	
	with open(path, "r", encoding="utf-8", errors="ignore") as f:
		lines= [l.strip() for l in f if l.strip()]
	
	result["instruction_count"]	= len(lines)
	
	mnemonics = []
	invalid_count=0
	
	for line in lines:
		opcode = line.split()[0]
		mnemonics.append(opcode)
		
		if opcode.lower() in ("invalid","unknown","trap"):
			invalid_count+=1
	result["unique_instructions"]=len(set(mnemonics))
	
	if len(lines)< MIN_INSTRUCTIONS:
		result["status"]="TOO_SHORT"
		result["reason"]="Insufficient instructions"
	
	elif result["unique_instructions"]<	MIN_UNIQUE_INSTRUCTIONS:
		result["status"]="LOW_VARIETY"
		result["reason"]="Very few unique instructions"
	return result
	
report = []
instruction_counts= []

for file in sorted(os.listdir(ASSEMBLY_DIR)):
	if not file.endswith(".asm"):
		continue
	path= os.path.join(ASSEMBLY_DIR,file)
	report.append(validate_file(path))
	
	if report[-1]["status"] == "VALID":
		instruction_counts.append(report[-1]["instruction_count"])

with open(REPORT, "w", newline="") as csvfile:
	writer= csv.DictWriter(csvfile, fieldnames=report[0].keys())
	writer.writeheader()
	writer.writerows(report)
	
total=len(report)
valid=sum(r["status"] == "VALID" for r in report)
short=sum(r["status"] == "TOO_SHORT" for r in report)
low_variety=sum(r["status"] == "LOW_VARIETY" for r in report)
empty = sum(r["status"] == "EMPTY" for r in report)

print(f"Total Files              : {total}")
print(f"Valid                    : {valid}")
print(f"Too Short                : {short}")
print(f"Low Variety              : {low_variety}")
print(f"Empty                    : {empty}")


if instruction_counts:
	print(f"Average Instructions     : {statistics.mean(instruction_counts):.1f}")
	print(f"Median Instructions      : {statistics.median(instruction_counts):.1f}")
	print(f"Minimum Instructions     : {min(instruction_counts)}")
	print(f"Maximum Instructions     : {max(instruction_counts)}")

