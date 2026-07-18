import os
import time
import r2pipe
import logging

logging.basicConfig(
	level= logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s',
	handlers = [
		logging.FileHandler("disassembly_benchmark.log"),
		logging.StreamHandler()
	]
)
RAW_DIR="data/raw/benign_mips"
OUTPUT_DIR="data/processed_assembly/benign_mips"
	
def extract_assembly(file_path,output_path):
	start_time=time.time()
	try:
		absolute_path = os.path.abspath(file_path)
		r2=r2pipe.open(absolute_path, flags=['-e','bin.relocs.apply=true'])
		r2.cmd('aaa')
		asm_out= r2.cmd('pda')
		
		if not asm_out:
			raise ValueError("No assembly extracted")
		cleaned_lines=[]
		for line in asm_out.splitlines():
			line=line.strip()
			if not line:
				continue
			parts=line.split(maxsplit=2)
			if len(parts)>=3 and parts[0].startswith("0x"):
				instruction=parts[2]
				if ';' in instruction:
					instruction=instruction.split(';')[0].strip()
				cleaned_lines.append(instruction)
		asm_out="\n".join(cleaned_lines)
		with open(output_path,'w') as f:
			f.write(asm_out)
			
		r2.quit()
		end_time=time.time()
		return True, (end_time-start_time)
	except Exception as e:
		return False, str(e)
		

binaries= [f for f in os.listdir(RAW_DIR) if os.path.isfile(os.path.join(RAW_DIR,f))]
logging.info(f"Starting bulk disassembly of {len(binaries)} files in {RAW_DIR}")
	
success=0
failure=0
tot_time=0.0
	
for filename in binaries:
	file_path= os.path.join(RAW_DIR,filename)
	output_path= os.path.join(OUTPUT_DIR,f"{filename}.asm")
		
	val,res=extract_assembly(file_path,output_path)
	if val:
		success+=1
		tot_time+=res
		logging.info(f"[SUCCESS] {filename} - Extracted in {res:.4f} seconds")
		print(f"[SUCCESS] {filename} - Extracted in {res:.4f} seconds")
	else:
		failure+=1
		logging.error(f"[FAILED] {filename} - Error: {res}")
		print(f"[FAILED] {filename} - Error: {res}")

avg_time= tot_time/success if success>0 else 0
logging.info("=== DISASSEMBLY BENCHMARK REPORT ===")
logging.info(f"Successful Extractions: {success}")
logging.info(f"Failed Extractions: {failure}")
logging.info(f"Average Time per Binary: {avg_time:.4f} seconds")
logging.info(f"Total Processing Time: {tot_time:.4f} seconds")

print("=== DISASSEMBLY BENCHMARK REPORT ===")
print(f"Successful Extractions: {success}")
print(f"Failed Extractions: {failure}")
print(f"Average Time per Binary: {avg_time:.4f} seconds")
print(f"Total Processing Time: {tot_time:.4f} seconds")

