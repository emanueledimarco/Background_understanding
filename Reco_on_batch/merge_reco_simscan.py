#!/usr/bin/env python
from os import system,access,F_OK,popen
from pathlib import Path
import sys
import os, re

# trick to import the function from the submission script in reconstruction
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "reconstruction"))
from scripts.submit_condor_recosim import find_files_with_dirs

def printAndExec(cmd,onlyprint=False):
    print(cmd)
    if not onlyprint:
        result = popen(cmd).read()
        print(result)


def mergeOneDir(inputdir,outputname='reco_merged.root',dryrun=False):

    print("Start merging...")
    files = find_files_with_dirs(inputdir,".root")

    outputfn = outputname
    result = sorted(files)
    print(f"===> List of {len(result)} files to be hadded:")
    print(result)
    print("================================")
    
    if access(outputfn,F_OK):
        print("skipping",outputfn)
        exit(0)
    if len(result) > 24:
        system(f"mkdir -p {inputdir}/Chunks")
        filesperintermediate = int((len(result))**0.5)
        subres = []
        nextone = 0
        while nextone < len(result):
            subres.append(result[nextone:nextone+filesperintermediate])
            nextone += filesperintermediate
        mediumlist = []    
        for i in range(len(subres)):
            mediumfile = outputfn.replace(".root",f"{inputdir}/intermediate{i}.root")
            mediumlist.append(mediumfile)
            if access(mediumfile,F_OK):
                print("skipping",mediumfile)
                continue
            cmd = "hadd %s %s" % (f"{inputdir}/{mediumfile}"," ".join([f"{fnn}" for fnn in subres[i]]))
            printAndExec(cmd,dryrun)
        cmd = "hadd %s %s" % (f"{outputfn}"," ".join(mediumlist)) 
        printAndExec(cmd,dryrun)
        print("Now moving all the chunks files in Chunks and remove the intermediate files...")
        system("mv %s {inputdir}/Chunks" % " ".join([fnn for fnn in result]))
        system("rm {inputdir}/*intermediate*root")
    else:    
        cmd = "hadd %s %s" % (f"{outputfn}"," ".join([f"{inputdir}/{fnn}" for fnn in result]))
        printAndExec(cmd,dryrun)
    
    
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("inputdir", help="base directory where the reconstructed files are (nested: path/subpath1/...histograms.root")
    parser.add_argument('--outputname', default="merged.root", type=str, help='output file name')
    parser.add_argument('--onedir', action="store_true", help='only merge one directory, do not look digi scan substructure')
    parser.add_argument('-d', '--dry_run', action="store_true", help='only print the merge commands, do not execute')
    args = parser.parse_args()

    if args.onedir:
        mergeOneDir(args.inputdir,args.outputname,args.dry_run)

    else:
        base_dir = args.inputdir
        pattern = re.compile(r"^digi_\d+-\d+$")

        target_dirs = []

        for root, dirs, files in os.walk(base_dir):
            # Trova tutte le subdirectory nel livello corrente che corrispondono al pattern
            matched_dirs = [d for d in dirs if pattern.match(d)]

            for d in matched_dirs:
                target_dirs.append(d)
                # Rimuove la directory trovata dalla lista 'dirs' di os.walk.
                # Questo impedisce a Python di entrare e scansionare il contenuto
                # della cartella digi_x-y, fermando la ricorsione a questo livello.
                dirs.remove(d)
                
        for path in target_dirs:
            print(path)
        print("I will merge the above results, separately, of all these directories, assuming uniform content inside each of them")

        mergedir = f"{base_dir}/merged"
        os.system(f"mkdir -p {mergedir}")
        for path in target_dirs:
            outputname = f"{mergedir}/{path}.root"
            print (f"{base_dir}/{path}")
            mergeOneDir(f"{base_dir}/{path}",outputname,args.dry_run)
