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
RAW_DIR=["data/raw/firmware","data/raw/malware_dataset"]
OUTPUT_DIR=["data/processed_assembly/benign_mips","data/processed_assembly/malware_mips"]
	
def extract_assembly(file_path,output_path):
	start_time=time.time()
	try:
		absolute_path = os.path.abspath(file_path)
		r2=r2pipe.open(absolute_path, flags=['-e','bin.relocs.apply=true'])
		r2.cmd('aaa')
		asm_out= r2.cmd('pda')
		
		functions= r2.cmdj("aflj")
		
		if not functions:
			raise ValueError("No functions extracted")
		instructions=[]
		
		for func in functions:
			offset=func["offset"]
			r2.cmd(f"s {offset}")
			pdf= r2.cmdj("pdfj")
			if not pdf:
				continue
			ops= pdf.get("ops",[])
			for op in ops:
				if "opcode" not in op:
					continue
				opcode=op["opcode"]
				opcode=opcode.split(';')[0].strip()
				if opcode:
					instructions.append(opcode)
		if len(instructions)==0:
			raise ValueError("No instruction extracted")
		with open(output_path,'w') as f:
			f.write("\n".join(instructions))
			
		r2.quit()
		end_time=time.time()
		return True, (end_time-start_time)
	except Exception as e:
		return False, str(e)
		

for i in range(2):	
	binaries= [f for f in os.listdir(RAW_DIR[i]) if os.path.isfile(os.path.join(RAW_DIR[i],f))]
	logging.info(f"Starting bulk disassembly of {len(binaries)} files in {RAW_DIR[i]}")
		
	success=0
	failure=0
	tot_time=0.0


	for filename in binaries:
		file_path= os.path.join(RAW_DIR[i],filename)
		output_path= os.path.join(OUTPUT_DIR[i],f"{filename}.asm")
			
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

