import argparse
import datetime
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def parse_submit_file_for_presign_json(submit_file_path):
    """Analizza il file .condor e restituisce la lista di percorsi dei file presign_*.json

    elencati nelle direttive transfer_input_files.
    """
    presign_files = []
    if not os.path.exists(submit_file_path):
        print(f"[WARN] File di submit non trovato: {submit_file_path}")
        return presign_files

    with open(submit_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("transfer_input_files"):
                # Cerca i file presign_*.json presenti nella riga
                matches = re.findall(
                    r"(\S+presign_[^\s,]+\.json)", line, re.IGNORECASE
                )
                for file_path in matches:
                    presign_files.append(Path(file_path.strip()))
    return presign_files


def update_presign_file(json_path):
    """Aggiorna il timestamp UTC (X-Amz-Date) all'interno di un singolo file presign.json."""
    if not json_path.exists():
        print(f"[WARN] File {json_path} non trovato.")
        return

    now_utc = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Aggiorna il timestamp mantenendo intatto il resto del link presigned
        updated_content = re.sub(
            r"X-Amz-Date=\d{8}T\d{6}Z", f"X-Amz-Date={now_utc}", content
        )

        with open(json_path, "w", encoding="utf-8") as f:
            f.write(updated_content)

        print(f"  [OK] Updated {json_path.name} -> {now_utc}")
    except Exception as e:
        print(f"  [ERROR] Errore su {json_path}: {e}")


def parse_commands(script_path,dry_run):
    """Estrae i comandi cygno_htc dal file condor_submit_all.sh."""
    commands = []
    if not os.path.exists(script_path):
        print(f"[ERROR] File {script_path} non trovato.")
        sys.exit(1)

    with open(script_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("cygno_htc"):
                if dry_run:
                    commands.append(f'echo "{line}"')
                else:
                    commands.append(line)
    return commands


def extract_submit_file_path(cmd):
    """Estrae il percorso del file .condor dal comando cygno_htc."""
    match = re.search(r"-s\s+(\S+)", cmd)
    if match:
        return Path(match.group(1))
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Gestisce la sottomissione a blocchi dei job HTCondor aggiornando i token presign.json."
    )
    parser.add_argument(
        "-s",
        "--script",
        default="condor_submit_all.sh",
        help="Percorso dello script shell contenente i comandi di sottomissione (default: condor_submit_all.sh)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=5,
        help="Numero N di comandi da inviare per ciascun blocco (default: 5)",
    )
    parser.add_argument(
        "-w",
        "--wait-minutes",
        type=float,
        default=10.0,
        help="Tempo M di attesa in minuti tra un blocco e il successivo (default: 10)",
    )
    parser.add_argument(
        '-d',
        '--dry-run',
        action="store_true",
        help='only print the merge commands, do not execute'
    )

    args = parser.parse_args()

    commands = parse_commands(args.script,args.dry_run)
    total_jobs = len(commands)

    if total_jobs == 0:
        print("Nessun comando cygno_htc trovato nel file.")
        return

    print(
        f"Trovati {total_jobs} comandi in {args.script}."
    )
    print(
        f"Esecuzione a blocchi di {args.batch_size} job con pausa di {args.wait_minutes} minuti tra i blocchi."
    )

    for i in range(0, total_jobs, args.batch_size):
        batch = commands[i : i + args.batch_size]
        batch_num = (i // args.batch_size) + 1
        total_batches = (total_jobs + args.batch_size - 1) // args.batch_size

        print(
            f"\n=== ESECUZIONE BLOCCO {batch_num}/{total_batches} (Jobs {i+1} - {min(i+args.batch_size, total_jobs)}) ==="
        )

        for cmd in batch:
            submit_file = extract_submit_file_path(cmd)
            if submit_file:
                print(
                    f"\n--- Analisi {submit_file.name} in {submit_file.parent.name} ---"
                )

                # 1. Trova ed aggiorna SOLO i file json dichiarati in questo submit.condor
                presign_files = parse_submit_file_for_presign_json(submit_file)
                if presign_files:
                    for p_file in presign_files:
                        update_presign_file(p_file)
                else:
                    print(
                        "  Nessun file presign_*.json trovato nel file di submit."
                    )

            # 2. Sottomissione del job
            bash_cmd = (
                f'bash -c "source $CVMFS_PARENT_DIR/cvmfs/sft-cygno.infn.it/config/cygno_htc '
                f'{cmd.partition("cygno_htc")[2]}"'
            )
            
            print(f"Esecuzione: {bash_cmd}")
            try:
                result = subprocess.run(
                    bash_cmd,
                    shell=True,
                    check=True,
                    text=True,
                    capture_output=True,
                )
                print(f"Output: {result.stdout.strip()}")
            except subprocess.CalledProcessError as e:
                print(
                    f"[ERROR] Fallita sottomissione per: {cmd}\nStdErr: {e.stderr}"
                )
                                                        
        # 3. Pausa M minuti tra un blocco e il successivo
        if i + args.batch_size < total_jobs:
            print(
                f"\nIn attesa di {args.wait_minutes} minuti prima del prossimo blocco..."
            )
            time.sleep(args.wait_minutes * 60)

    print("\nTutti i blocchi sono stati completati.")


if __name__ == "__main__":
    main()
