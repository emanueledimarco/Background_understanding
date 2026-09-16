#!/usr/bin/env python
import argparse
import re
import subprocess
import sys
import time


def get_active_job_ids(ce):
    """Esegue 'cygno_htc -q <CE>' ed estrae gli ID numerici dei cluster/job."""
    cmd = f'bash -c "source $CVMFS_PARENT_DIR/cvmfs/sft-cygno.infn.it/config/cygno_htc -q {ce}"'
    job_ids = []

    try:
        result = subprocess.run(
            cmd, shell=True, check=True, text=True, capture_output=True
        )
        lines = result.stdout.splitlines()

        for line in lines:
            line = line.strip()
            # Salta intestazioni e righe vuote
            if not line or line.startswith("--") or line.startswith("OWNER"):
                continue

            # Cerca il pattern 'ID: <JOB_ID>' presente nell'output di condor
            match = re.search(r"ID:\s*(\d+)", line)
            if match:
                job_ids.append(match.group(1))

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Impossibile recuperare la coda per CE {ce}: {e.stderr.strip()}")

    return job_ids


def apply_action_to_job(job_id, ce, action, dry_run):
    """Esegue l'azione (-f o -r) su un determinato job ID e CE."""
    cmd_str = f"source $CVMFS_PARENT_DIR/cvmfs/sft-cygno.infn.it/config/cygno_htc {action} {job_id} {ce}"
    bash_cmd = f'bash -c "{cmd_str}"'

    print(f"  [ACTION] Esecuzione: cygno_htc {action} {job_id} {ce}")

    if not dry_run:
        try:
            res = subprocess.run(
                bash_cmd, shell=True, check=True, text=True, capture_output=True
            )
            out = res.stdout.strip()
            if out:
                print(f"    [OUTPUT] {out}")
        except subprocess.CalledProcessError as e:
            print(f"    [ERROR] Fallito su job {job_id} (CE {ce}): {e.stderr.strip()}")


def run_cycle(ce_list, action, dry_run):
    """Effettua un singolo ciclo di scasione su tutti i CE configurati."""
    total_jobs_found = 0

    for ce in ce_list:
        print(f"\n--- Controllo Computing Element (CE) {ce} ---")
        job_ids = get_active_job_ids(ce)

        if not job_ids:
            print(f"  Nessun job trovato per CE {ce}.")
            continue

        print(f"  Trovati {len(job_ids)} job su CE {ce}: {', '.join(job_ids)}")
        total_jobs_found += len(job_ids)

        for jid in job_ids:
            apply_action_to_job(jid, ce, action, dry_run)

    return total_jobs_found


def main():
    parser = argparse.ArgumentParser(
        description="Gestore loop sui CE di HTCondor per azione batch (es. -f o -r) sui job attivi.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ce-start",
        type=int,
        default=1,
        help="Primo Computing Element del range",
    )
    parser.add_argument(
        "--ce-end",
        type=int,
        default=5,
        help="Ultimo Computing Element del range (incluso)",
    )
    parser.add_argument(
        "-a",
        "--action",
        type=str,
        default="-f",
        choices=["-f", "-r"],
        help="Azione da applicare sul job (es. -f per force/remove, -r per release/hold)",
    )
    parser.add_argument(
        "-N",
        "--wait-minutes",
        type=float,
        default=0.0,
        help="Minuti di attesa tra un ciclo e il successivo (se 0 fa un solo giro ed esce)",
    )
    parser.add_argument(
        "-M",
        "--max-cycles",
        type=int,
        default=1,
        help="Numero massimo di cicli da eseguire prima di uscire",
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="Mostra soltanto i comandi che verrebbero eseguiti senza lanciarli davvero",
    )

    args = parser.parse_args()

    ce_list = list(range(args.ce_start, args.ce_end + 1))
    current_cycle = 1

    print("=== CYGNO HTCondor Job Manager ===")
    print(f"CE Range: {ce_list}")
    print(f"Azione selezionata: {args.action}")
    print(f"Intervallo attesa (N): {args.wait_minutes} min")
    print(f"Numero massimo cicli (M): {args.max_cycles}")
    if args.dry_run:
        print("[DRY RUN ATTIVO - Nessuna modifica verrà applicata]")

    while True:
        print(f"\n=======================================================")
        print(f"  INIZIO CICLO {current_cycle}/{args.max_cycles}")
        print(f"=======================================================")

        jobs_processed = run_cycle(ce_list, args.action, args.dry_run)
        print(f"\n[Sommario Ciclo {current_cycle}] Processati {jobs_processed} job totali.")

        # Condizione di uscita 1: Raggiunto il limite massimo di cicli M
        if current_cycle >= args.max_cycles:
            print("\nRaggiunto il numero massimo di cicli impostato (M). Esecuzione terminata.")
            break

        # Condizione di uscita 2: Intervallo N impostato a 0
        if args.wait_minutes <= 0:
            print("\nTempo di attesa (N) uguale a 0. Esecuzione terminata dopo un singolo ciclo.")
            break

        # Pausa tra i cicli
        print(f"\n[WAIT] In attesa di {args.wait_minutes} minuti prima del prossimo ciclo...")
        time.sleep(args.wait_minutes * 60)
        current_cycle += 1


if __name__ == "__main__":
    main()

