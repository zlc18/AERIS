#!/usr/bin/env bash
# Run the full revision pipeline in dependency order; stop at the first failure.
# Usage (from the repository root): bash scripts/revision/run_pipeline.sh [first-step]
set -euo pipefail
cd "$(dirname "$0")/../.."
LOG_DIR=outputs/revision/logs
mkdir -p "$LOG_DIR"

STEPS=(
  "rev00_build_feature_table"
  "rev01_label_validation"
  "rev01b_score_structure"
  "rev08_feature_dictionary"
  "rev02_baselines"
  "rev09_baseline_search --trials 16"
  "rev09b_regression_baselines --trials 16"
  "rev10_aeris_search --trials 12 --seeds 42,7,13"
  "rev10b_finalise_aeris --seeds 42,7,13 --select-seeds 1"
  "rev10c_multitask_search --trials 14 --seeds 42,7,13"
  "rev17_feature_selection --seeds 42,7,13"
  "rev16_ensembles"
  "rev22_dualbranch_search --trials 10 --seeds 42,7,13"
  "rev23_final_selection --seeds 42,7 --final-seeds 42,7,13"
  "rev28_deploy_best"
  "rev03_sequence_models --seeds 42,7,13 --epochs 8"
  "rev04_ablation --seeds 42,7 --sweep-seeds 42 --epochs 12"
  "rev05_robustness --parts ABCDEF --seeds 42"
  "rev06_kde_confidence"
  "rev06b_kde_confidence"
  "rev21_horizon_retune --seeds 42,7 --horizons 1,2,3,4,5,6,8"
  "rev27_horizon_significance --seeds 42,7,13 --horizons 1,2,3,4,5,6,8"
  "rev30_horizon_vs_persistence"
  "rev33_horizon_lightgbm"
  "rev31_block_sensitivity"
  "rev26_calibrate"
  "rev07_evaluation"
  "rev32_audit_tuned_regressors"
  "rev32b_quantiles_tuned_regressor"
  "rev32c_audit_tuned_classifier"
  "rev11_extra_figures"
  "rev34_schematic_figures"
  "rev12_make_tables"
  "rev14_report"
)

started=${1:-}
for step in "${STEPS[@]}"; do
  name=${step%% *}
  if [[ -n "$started" && "$name" != "$started" ]]; then continue; fi
  started=""
  echo "[$(date '+%H:%M:%S')] START $step"
  if ! python -m scripts.revision.$step > "$LOG_DIR/$name.log" 2>&1; then
    echo "[$(date '+%H:%M:%S')] FAILED $name (see $LOG_DIR/$name.log)"
    tail -20 "$LOG_DIR/$name.log"
    exit 1
  fi
  echo "[$(date '+%H:%M:%S')] DONE  $name"
done
echo "[$(date '+%H:%M:%S')] PIPELINE COMPLETE"
