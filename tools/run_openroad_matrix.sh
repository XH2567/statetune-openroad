#!/bin/bash
# Run the formal StateTune/OpenROAD experiments for all 6 (PDK x design) cases,
# at most $PARALLEL at a time. Detach with:  setsid nohup tools/run_openroad_matrix.sh > /dev/null 2>&1 &
# Progress: runs/matrix_status.txt ; per-case log: runs/<project_name>.log
PARALLEL=${PARALLEL:-3}
MIN_FREE_GB=${MIN_FREE_GB:-40}   # do not start a case unless the ORFS disk has this much free
CASES=${CASES:-"nangate_jpeg asap7_ibex asap7_jpeg nangate_ibex asap7_aes nangate_aes"}

cd "$(dirname "$0")/.." && source .env
: "${ORFS_HOME:?set ORFS_HOME in .env}"
STATUS=runs/matrix_status.txt
echo "START $(date '+%F %T') parallel=$PARALLEL cases=[$CASES]" >> $STATUS

run_case() {
  local c=$1
  local name avail
  name=$(python -c "import yaml;print(yaml.safe_load(open('configs/$c/openroad_tuner.yaml'))['project_name'])")
  # ORFS writes ~0.4-0.8 GB per run under $ORFS_HOME; a shared disk filled up once mid-run,
  # so wait for headroom instead of launching into a full disk
  while avail=$(df --output=avail -BG "$ORFS_HOME" | tail -1 | tr -dc '0-9'); [ "$avail" -lt "$MIN_FREE_GB" ]; do
    echo "WAIT  $c: $ORFS_HOME disk has ${avail}G free (< ${MIN_FREE_GB}G) $(date '+%F %T')" >> $STATUS
    sleep 600
  done
  echo "BEGIN $c ($name) free=${avail}G $(date '+%F %T')" >> $STATUS
  python -m src.EDATuner.main --config configs/$c/openroad_tuner.yaml > runs/$name.log 2>&1
  echo "END   $c rc=$? $(date '+%F %T')" >> $STATUS
}
export -f run_case
export STATUS MIN_FREE_GB ORFS_HOME
printf "%s\n" $CASES | xargs -P "$PARALLEL" -I{} bash -c 'run_case {}'
echo "ALL DONE $(date '+%F %T')" >> $STATUS
