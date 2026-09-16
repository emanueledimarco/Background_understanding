#!/usr/bin/env python
# USAGE: ./submit_trickle_digi.py -a 0.021 -l 1350 -o out_giulia_cu -i /cnaf/cygno-sim/Users/dimarcoe/digitune/digi_giulia_cu -s users/dimarcoe/digi/cu_giulia $PWD

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ENDPOINT_URL = "https://s3.cr.cnaf.infn.it:7480/"

jobstring = """#!/bin/bash
ulimit -c 0 -S
ulimit -c 0 -H
set -e

# Experiment executable config
export CVMFS_PARENT_DIR=""
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-ubuntu2204-gcc11-opt/setup.sh
source /cvmfs/sft-cygno.infn.it/config/setup_digi.sh
"""


def makeInputList(inputdir):
    inputcloud = re.sub(r"^.*?(?=cygno-)", "", inputdir)
    inputcloud = os.path.normpath(inputcloud)
    full_url = f"{ENDPOINT_URL}cygno:{inputcloud}"
    full_url = re.sub(
        r"(?<!:)/{2,}", "/", full_url
    )  # remove eventual last // which prevents wget from cloud
    wget_cmds = []
    for file in Path(inputdir).glob("*.root"):
        if file.is_file():
            wget_cmds.append(f"wget {full_url}/{file.name}")
    return wget_cmds


def makePreSign(jobdir, jobnumber, outfile, options):
    """Genera e/o aggiorna in tempo reale il file presign.json prima della sottomissione."""
    BUCKET = options.bucket
    TAG = f"{options.storagedir}/{Path(jobdir).name}/job_{jobnumber}"
    FILETOKEN = "/tmp/token"

    cmd = f"/cvmfs/sft-cygno.infn.it/config/lib/presigned.py -u {ENDPOINT_URL} -b {BUCKET} -t {TAG} {outfile} -f {FILETOKEN} > {jobdir}/presign_job{jobnumber}.json"
    print(
        f"  [PRESIGN] Rigenerazione URL presigned per job_{jobnumber}..."
    )
    os.system(cmd)


def makeCondorFile(condor_file_name, jobdir, srcFiles, cfgFile, options):
    dummy_exec = open(jobdir + "/dummy_exec.sh", "w")
    dummy_exec.write("#!/bin/bash\n")
    dummy_exec.write("bash $*\n")
    dummy_exec.close()

    condor_file = open(condor_file_name, "w")
    condor_file.write(
        """+SingularityImage = "/cvmfs/sft-cygno.infn.it/dockers/images/cygno-wn_v2.4.sif"
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
""".format(
            de=dummy_exec.name,
            ld=os.path.abspath(jobdir),
            od=os.path.abspath(jobdir),
            ed=os.path.abspath(jobdir),
            cpu=options.threads,
            user=os.environ.get("USER", os.environ.get("USERNAME", "cygno")),
            here=os.environ["PWD"],
        )
    )
    for isrc, src in enumerate(srcFiles):
        condor_file.write(
            f"transfer_input_files = {os.path.abspath(options.srcdir)}/build-dir, {os.path.abspath(options.srcdir)}/VignettingMap, {os.path.abspath(options.srcdir)}/pmt, {os.path.abspath(cfgFile)}, {os.path.abspath(src)}, /cvmfs/sft-cygno.infn.it/config/lib/s3upload_put.py, {jobdir}/presign_job{isrc}.json\n"
        )
        condor_file.write(f"arguments = {os.path.basename(src)} \nqueue \n\n")

    condor_file.close()


def replaceParam(input_file, old_string, new_string, output_file=None):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    new_lines = [line.replace(old_string, new_string) for line in lines]

    if not output_file:
        output_file = input_file
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def submit_condor_job(condor_file_path, ce, dry_run):
    """Esegue il comando cygno_htc risolvendo l'alias tramite la source diretta da CVMFS."""
    cmd = (
        f'bash -c "source $CVMFS_PARENT_DIR/cvmfs/sft-cygno.infn.it/config/cygno_htc '
        f'-s {condor_file_path} {ce}"'
    )
    print(f"  [SUBMIT] Invocazione: {cmd}")

    if dry_run == False:
        try:
            result = subprocess.run(
                cmd, shell=True, check=True, text=True, capture_output=True
            )
            print(f"  [OUTPUT] {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(
                f"  [ERROR] Fallita sottomissione per {condor_file_path}:\n{e.stderr}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("srcdir", help="base directory where build-dir is")
    parser.add_argument(
        "-a",
        "--alphas",
        type=float,
        nargs="*",
        default=np.linspace(0.019, 0.023, 11),
        help="List of alpha values to scan",
    )
    parser.add_argument(
        "-l",
        "--lambdas",
        type=float,
        nargs="*",
        default=np.linspace(850, 1850, 11),
        help="List of absorption length values (in mm) to scan",
    )
    parser.add_argument(
        "-C",
        "--CE",
        type=int,
        default=2,
        help="Computing element in condor to use",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=8,
        help="Number of CPUs to request",
    )
    parser.add_argument(
        "-o", "--outdir", type=str, default=None, help="output directory"
    )
    parser.add_argument(
        "-i", "--inputdir", type=str, default=None, help="input directory"
    )
    parser.add_argument(
        "-b",
        "--bucket",
        type=str,
        default="cygno-analysis",
        help="bucket in the cloud where to store the output",
    )
    parser.add_argument(
        "-s",
        "--storagedir",
        type=str,
        default="users/dimarcoe/digi/fe_zcone",
        help="output directory in the cloud",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config/ConfigFile_new.txt",
        help="config file for DIGI to be used",
    )

    # Parametri aggiuntivi per il flusso a blocchi con wait
    parser.add_argument(
        "-B",
        "--batch-size",
        type=int,
        default=4,
        help="Numero di blocchi (file submit.condor) da sottomettere prima della pausa",
    )
    parser.add_argument(
        "-W",
        "--wait-minutes",
        type=float,
        default=30.0,
        help="Tempo di attesa in minuti tra una sottomissione a blocchi e la successiva",
    )

    # Parametro per specificare da quale directory riprendere la sottomissione
    parser.add_argument(
        "-f",
        "--start-from",
        type=str,
        default=None,
        help="Pattern della directory da cui ripartire (es. '1_2', '1-2' o 'digi_1-2')",
    )

    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="only print the merge commands, do not execute",
    )

    args = parser.parse_args()

    print("=== SUBMIT DIGI (Sottomissione a Blocchi con Presign Live) ===")
    absopath = os.path.abspath(args.outdir)
    if not args.outdir:
        raise RuntimeError("ERROR: give at least an output directory.")
    else:
        if not os.path.isdir(absopath):
            print("making a directory and running in it")
            os.system(f"mkdir -p {absopath}")

    if not args.inputdir.startswith("/cnaf/cygno-"):
        raise RuntimeError(
            'ERROR: inputdir should start with "/cnaf/cygno-"'
        )

    input_wget_cmds = makeInputList(args.inputdir)
    if len(input_wget_cmds) == 0:
        raise RuntimeError(
            f"ERROR: no input ROOT files found in {args.inputdir}. Exit."
        )

    print(
        f"==> Each condor cluster will run on {len(input_wget_cmds)} ROOT files"
    )

    jobdir = os.path.join(absopath, "jobs")
    if not os.path.isdir(jobdir):
        os.system(f"mkdir -m 777 -p {jobdir}")

    # Costruiamo l'elenco di tutte le combinazioni di parametri
    param_grid = [
        (a, alpha, l, Lambda)
        for a, alpha in enumerate(args.alphas)
        for l, Lambda in enumerate(args.lambdas)
    ]

    # Gestione della partenza da uno specifico punto (--start-from)
    if args.start_from:
        target_indices = re.sub(r"^digi_", "", args.start_from)
        target_indices = target_indices.replace("_", "-")

        try:
            target_a, target_l = map(int, target_indices.split("-"))
        except ValueError:
            raise RuntimeError(
                f"ERROR: Formato non valido per --start-from '{args.start_from}'. "
                f"Usa formati come '1_2', '1-2' oppure 'digi_1-2'."
            )

        start_index = None
        for idx, (a, alpha, l, Lambda) in enumerate(param_grid):
            if a == target_a and l == target_l:
                start_index = idx
                break

        if start_index is None:
            raise RuntimeError(
                f"ERROR: La combinazione di indici ({target_a}, {target_l}) "
                f"non è presente nei valori di alpha e lambda correnti."
            )

        param_grid = param_grid[start_index:]
        print(
            f"[INFO] Ripresa sottomissione a partire da digi_{target_a}-{target_l} (index {start_index})"
        )

    total_tasks = len(param_grid)
    print(
        f"Totale configurazioni da inviare: {total_tasks} a blocchi di {args.batch_size}"
    )

    # Ciclo di creazione e sottomissione a blocchi
    for idx in range(0, total_tasks, args.batch_size):
        batch = param_grid[idx : idx + args.batch_size]
        batch_num = (idx // args.batch_size) + 1
        total_batches = (total_tasks + args.batch_size - 1) // args.batch_size

        print(
            f"\n======================================================="
        )
        print(
            f"  INIZIO BLOCCO {batch_num}/{total_batches} (Job {idx+1} -> {min(idx+args.batch_size, total_tasks)})"
        )
        print(
            f"======================================================="
        )

        for a, alpha, l, Lambda in batch:
            print(f"\n---> Preparazione per (alpha={alpha:.3f}, Lambda={Lambda:.0f})")
            con_file_name = f"{jobdir}/conf_{a}-{l}.txt"
            os.system(f"cp {args.srcdir}/{args.config} {con_file_name}")

            replaceParam(
                con_file_name,
                "'queue'                 : 0",
                "'queue'                 : 1",
            )
            replaceParam(
                con_file_name,
                "'absorption_l'          : 1350.",
                f"'absorption_l'          : {Lambda:.0f}",
            )
            replaceParam(
                con_file_name,
                "'alpha_G'               : 0.0209",
                f"'alpha_G'               : {alpha:.3f}",
            )

            ijobdir = f"{jobdir}/digi_{a}-{l}"
            os.system(f"mkdir -m 777 -p {ijobdir}")

            cmd = f"\n./build-dir/digitizationpp ./{os.path.basename(con_file_name)} -I ./ -O ./"

            srcfiles = []
            for iw, wget in enumerate(input_wget_cmds):
                job_file_name = f"{ijobdir}/job_{a}-{l}_job{iw}.sh"
                outfile_prefix = "histograms_Run00001"

                with open(job_file_name, "w") as tmp_file:
                    tmp_filecont = jobstring
                    tmp_filecont += f"\n{wget}"
                    tmp_filecont += cmd
                    tmp_filecont += f"\n./s3upload_put.py presign_job{iw}.json"
                    tmp_filecont += "\necho DONE.\n"
                    tmp_file.write(tmp_filecont)

                # Genera il file presign FRESCO immediatamente prima del submit
                makePreSign(
                    ijobdir, iw, f"{outfile_prefix}.root", args
                )
                srcfiles.append(job_file_name)

            # Crea il file submit.condor per questo blocco specifico
            condor_fname = f"{ijobdir}/submit.condor"
            makeCondorFile(
                condor_fname, ijobdir, srcfiles, con_file_name, args
            )

            # Sottomette subito il file .condor tramite cygno_htc
            submit_condor_job(condor_fname, args.CE, args.dry_run)

        # Attesa tra i blocchi (tranne dopo l'ultimo)
        if idx + args.batch_size < total_tasks:
            print(
                f"\n[WAIT] In attesa di {args.wait_minutes} minuti prima di preparare e sottomettere il prossimo blocco..."
            )
            time.sleep(args.wait_minutes * 60)

    print("\nTutti i blocchi sono stati completati.")
    sys.exit(0)
