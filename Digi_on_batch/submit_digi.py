#!/usr/bin/env python
# USAGE: /test/submit_digi.py -a 0.021 -l 1350 -o out_giulia_cu -i /cnaf/cygno-sim/Users/dimarcoe/digitune/digi_giulia_cu -s users/dimarcoe/digi/cu_giulia $PWD
import os, sys, re
import numpy as np
from pathlib import Path

ENDPOINT_URL='https://s3.cr.cnaf.infn.it:7480/'

jobstring  = '''#!/bin/bash
ulimit -c 0 -S
ulimit -c 0 -H
set -e

# Experiment executable config
export CVMFS_PARENT_DIR=""
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-ubuntu2204-gcc11-opt/setup.sh
source /cvmfs/sft-cygno.infn.it/config/setup_digi.sh
'''

def makeInputList(inputdir):
    inputcloud = re.sub(r'^.*?(?=cygno-)', '', inputdir)
    inputcloud = os.path.normpath(inputcloud)
    full_url = f"{ENDPOINT_URL}cygno:{inputcloud}"
    full_url = re.sub(r'(?<!:)/{2,}', '/', full_url) # remove eventual last // wich prevents wget from cloud
    wget_cmds = []
    for file in Path(inputdir).glob("*.root"):
        if file.is_file():
            wget_cmds.append(f"wget {full_url}/{file.name}")
    return wget_cmds

def makePreSign(jobdir,jobnumber,outfile,options):
    BUCKET=options.bucket
    TAG=f'{options.storagedir}/{Path(jobdir).name}/job_{jobnumber}'
    FILETOKEN='/tmp/token'

    cmd = f'/cvmfs/sft-cygno.infn.it/config/lib/presigned.py -u {ENDPOINT_URL} -b {BUCKET} -t {TAG} {outfile} -f {FILETOKEN} > {jobdir}/presign_job{jobnumber}.json'
    print(f"generating presigned url for: with command: {cmd}")
    os.system(cmd)

def makeCondorFile(condor_file_name, jobdir, srcFiles, cfgFile, options):

    dummy_exec = open(jobdir+'/dummy_exec.sh','w')
    dummy_exec.write('#!/bin/bash\n')
    dummy_exec.write('bash $*\n')
    dummy_exec.close()
     
    condor_file = open(condor_file_name,'w')
    condor_file.write('''+SingularityImage = "/cvmfs/sft-cygno.infn.it/dockers/images/cygno-wn_v2.4.sif"
+SingularityBind = "/cvmfs/:/cvmfs/"
Requirements = HasSingularity

Executable = {de}
Log        = {ld}/$(ProcId).log
Output     = {od}/$(ProcId).out
Error      = {ed}/$(ProcId).error
getenv      = True
next_job_start_delay = 1
environment = "LS_SUBCWD={here}"
request_cpus = {cpu}
should_transfer_files   = YES
preserve_relative_paths = True
+CygnoUser = "{user}"\n
'''.format(de=dummy_exec.name,
           ld=os.path.abspath(jobdir), od=os.path.abspath(jobdir),ed=os.path.abspath(jobdir),
           cpu=options.threads, user=os.environ['USERNAME'], here=os.environ['PWD'] ) )
    for isrc,src in enumerate(srcFiles):
        print (f"isrcfile = {isrc}, sf={src}")
        condor_file.write(f'transfer_input_files = {os.path.abspath(options.srcdir)}/build-dir, {os.path.abspath(options.srcdir)}/VignettingMap, {os.path.abspath(cfgFile)}, {os.path.abspath(src)}, /cvmfs/sft-cygno.infn.it/config/lib/s3upload_put.py, {jobdir}/presign_job{isrc}.json\n')
        condor_file.write(f'arguments = {os.path.basename(src)} \nqueue \n\n')
        
    condor_file.close()

def replaceParam(input_file,old_string,new_string,output_file=None):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    new_lines = [line.replace(old_string, new_string) for line in lines]

    if not output_file:
        output_file = input_file
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"Replaced {old_string} with {new_string} in {output_file}")

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("srcdir", help="base directory where build-dir is")
    parser.add_argument("-a", "--alphas", type=float, nargs="*",  default=np.linspace(0.019,0.023,11), help="List of alpha values to scan")
    parser.add_argument("-l", "--lambdas", type=float, nargs="*",  default=np.linspace(850,1850,11), help="List of absorption length values (in mm) to scan")
    parser.add_argument("-C", "--CE", type=int, default=2, help="Computing element in condor to use")
    parser.add_argument("-t", "--threads", type=int, default=8, help="Number of CPUs to request")
    parser.add_argument("-o", "--outdir", type=str, default=None, help='output directory');
    parser.add_argument("-i", "--inputdir", type=str, default=None, help='input directory');
    parser.add_argument("-b", "--bucket", type=str, default="cygno-analysis", help='bucket in the cloud where to store the output');
    parser.add_argument("-s", "--storagedir", type=str, default="users/dimarcoe/digi/fe_zcone", help='output directory in the cloud');
    parser.add_argument("-c", "--config", type=str, default="config/ConfigFile_new.txt", help='config file for DIGI to be used');
    args = parser.parse_args()

    print("SUBMIT DIGI")
    absopath  = os.path.abspath(args.outdir)
    if not args.outdir:
        raise RuntimeError ('ERROR: give at least an output directory. there will be a HUGE number of jobs!')
    else:
        if not os.path.isdir(absopath):
            print ('making a directory and running in it')
            os.system('mkdir -p {od}'.format(od=absopath))

    if not args.inputdir.startswith("/cnaf/cygno-"):
        raise RuntimeError ('ERROR: inputdir should start with "/cnaf/cygno-" because the job copies the inputfiles from cloud with wget fro the 3 allowed cygno-<bucket>s')
    else:
        number_root_outfiles = sum(1 for file in Path(args.inputdir).glob("*.root") if file.is_file()) 
        print(f"==> Each condor cluster will run on {number_root_outfiles} ROOT files, 1 job / input file, and produce the same number of output files")
        
    jobdir = absopath+'/jobs/'
    if not os.path.isdir(jobdir):
        os.system('mkdir -m 777 -p {od}'.format(od=jobdir))

    condorfiles = []
    for a,alpha in enumerate(args.alphas):
        for l,Lambda in enumerate(args.lambdas):
            print (f"Prepare job for pair (alpha,Lambda) = ({alpha},{Lambda})")
            con_file_name = f"{jobdir}/conf_{a}-{l}.txt"
            os.system(f"cp {args.srcdir}/{args.config} {con_file_name}") 

            replaceParam(con_file_name,"'absorption_l'          : 1350.",  f"'absorption_l'          : {Lambda:.0f}")
            replaceParam(con_file_name,"'alpha_G'               : 0.0209", f"'alpha_G'               : {alpha:.3f}")
            #replaceParam(con_file_name,"'events'                : -1",f"'events'                : 10")
            
            ijobdir = f'{jobdir}/digi_{a}-{l}'
            os.system(f'mkdir -m 777 -p {ijobdir}')

            # N.B.: use explicitly "./" as the directory for the config file name, because the path for the vignetting is built from there in DigitizationRunner.cxx
            cmd = f"\n./build-dir/digitizationpp ./{os.path.basename(con_file_name)} -I ./ -O ./"

            input_wget_cmds = makeInputList(args.inputdir)
            if len(input_wget_cmds)==0:
                raise RuntimeError (f'ERROR: no input ROOT files found in {args.inputdir}. Exit.')
            # parallelize more: 1job/input file
            srcfiles=[]
            for iw,wget in enumerate(input_wget_cmds):
                job_file_name = f"{ijobdir}/job_{a}-{l}_job{iw}.sh"
                log_file_name = f"{ijobdir}/job_{a}-{l}_job{iw}.log"
                outfile_prefix = 'histograms_Run00001' # for 1 input file / job, this is always the name
                tmp_file = open(job_file_name, 'w')

                tmp_filecont = jobstring
                tmp_filecont += f'\n{wget}'
                tmp_filecont += cmd
                tmp_filecont += f"\n./s3upload_put.py presign_job{iw}.json"
                tmp_filecont += "\necho DONE.\n"
                tmp_file.write(tmp_filecont)
                tmp_file.close()
                makePreSign(ijobdir,iw,f'{outfile_prefix}.root',args)
                srcfiles.append(job_file_name)

            condor_fname = f'{ijobdir}/submit.condor'
            cf = makeCondorFile(condor_fname,ijobdir,srcfiles,con_file_name,args)
            condorfiles.append(condor_fname)
    print (f"Condor files:\n{condorfiles}")

    with open(f'{absopath}/condor_submit_all.sh','w') as sub_all:
        sub_all.write("#!/bin/bash\n\n")
        for con in condorfiles:
            sub_all.write(f'cygno_htc -s {con} {args.CE}\n')
    print (f"READY to submit. Now source {absopath}/condor_submit_all.sh")

    sys.exit()
    
