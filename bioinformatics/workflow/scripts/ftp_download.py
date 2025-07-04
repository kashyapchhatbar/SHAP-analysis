from ftplib import FTP
from tqdm.auto import tqdm
from operator import itemgetter
from subprocess import Popen, PIPE, run

mate_one_files = {}
mate_two_files = {}

with open(snakemake.input[0], 'r') as f:
    for line in f.readlines():
        if line.strip().endswith("_1.fastq.gz"):
            mate_one_files[line.strip().rpartition("/")[-1]] = line.strip()
        elif line.strip().endswith("_2.fastq.gz"):
            mate_two_files[line.strip().rpartition("/")[-1]] = line.strip()
        elif line.strip().endswith("fastq.gz"):
            mate_one_files[line.strip().rpartition("/")[-1]] = line.strip()
        
for cwd, file in mate_one_files.items():
    run(["wget", "-O", f"{snakemake.resources.tmpdir}/{cwd}", file], capture_output=True)

for cwd, file in mate_two_files.items():
    run(["wget", "-O", f"{snakemake.resources.tmpdir}/{cwd}", file], capture_output=True)
    
mate_one_file_list = [f"{snakemake.resources.tmpdir}/{v}" for v, _ in sorted(mate_one_files.items(), key=itemgetter(0))]
mate_two_file_list = [f"{snakemake.resources.tmpdir}/{v}" for v, _ in sorted(mate_two_files.items(), key=itemgetter(0))]


if len(mate_one_file_list) >= 1 and len(mate_two_file_list) >= 1:

    with open(snakemake.output[0], 'wb') as f:
        mate_one_process = Popen(['cat'] + mate_one_file_list, stdout=PIPE, stderr=PIPE)
        mate_one_stdout, mate_one_stderr = mate_one_process.communicate()
        f.write(mate_one_stdout)
        
    with open(snakemake.output[1], 'wb') as f:
        mate_two_process = Popen(['cat'] + mate_two_file_list, stdout=PIPE, stderr=PIPE)
        mate_two_stdout, mate_two_stderr = mate_two_process.communicate()
        f.write(mate_two_stdout)

if len(mate_one_file_list) >= 1:
    with open(snakemake.output[0], 'wb') as f:
        mate_one_process = Popen(['cat'] + mate_one_file_list, stdout=PIPE, stderr=PIPE)
        mate_one_stdout, mate_one_stderr = mate_one_process.communicate()
        f.write(mate_one_stdout)
        
    run(["touch", snakemake.output[1]])