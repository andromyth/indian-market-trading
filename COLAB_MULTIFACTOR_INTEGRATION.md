# Google Colab integration

The main branch is not replaced. This branch adds an additive candidate build
path.

In Colab:

git fetch origin
git checkout multifactor-readiness-v2
python colab_run_multifactor_upgrade.py

Google Drive remains the durable research-data and candidate-model store.
Colab builds and validates the candidate. The existing production model is not
overwritten and there is no automatic GitHub push or production promotion.

Candidate output:
Indian_Market_Research/models/all_nse_delivery_multifactor_candidate_2026/

Review DATA_READINESS_REPORT_2026.json and CANDIDATE_VALIDATION_REPORT_2026.json
before any production integration.
