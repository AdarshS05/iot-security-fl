import os
from pathlib import Path

INPUT_ROOT = Path("data/normalized_assembly")
OUTPUT_ROOT= Path("data/ngrams")

N= 2

def generate_ngrams(tokens, n):
	if len(tokens)<n:
		return []
		
	return [ "_".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
	
def process_file(input_path, output_path, n):
	tokens = []
	with open(input_path, "r", encoding="utf-8") as f:
		for line in f:
			line=line.strip()
			
			if not line:
				continue
				
			tokens.extend(line.split())
			
	ngrams=generate_ngrams(tokens,n)
	
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with open(output_path, "w", encoding="utf-8") as f:
		f.write("\n".join(ngrams))
		
	return len(tokens),len(ngrams)
	
def process_directory(input_dir, output_dir, n):
	total_files = 0
	total_tokens = 0
	total_ngrams = 0

	for file in sorted(input_dir.glob("*.asm")):
		output_file = output_dir / file.name
		tokens, ngrams = process_file(file, output_file, n)
		
		total_files += 1
		total_tokens += tokens
		total_ngrams += ngrams

	return total_files, total_tokens, total_ngrams

datasets = [("benign_mips", INPUT_ROOT / "benign_mips", OUTPUT_ROOT / "benign_mips"), ("malware_mips", INPUT_ROOT / "malware_mips", OUTPUT_ROOT / "malware_mips"),]

print(f"Generating {N}-grams")
print("=" * 50)

for name, input_dir, output_dir in datasets:
	files, tokens, ngrams = process_directory(input_dir,output_dir,N)
	
	print(f"\n{name}")
	print(f"Files Processed : {files}")
	print(f"Tokens          : {tokens:,}")
	print(f"{N}-grams       : {ngrams:,}")
	
	print("\nCompleted.")
