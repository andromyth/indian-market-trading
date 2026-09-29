# Colab → private GitHub → Streamlit

1. Google Drive is the system of record for historical NSE data.
2. Google Colab mounts the same Drive and runs `colab_build_nse_model.py`.
3. Colab writes only the small model release bundle:
   - `nse_production_candidate_30m.joblib`
   - `NSE_PRODUCTION_MODEL_MANIFEST_2026.json`
   - `MODEL_TARGET_SEMANTICS_TEST.txt`
4. Copy the release bundle into the private GitHub repository under
   `production_model/` only after the holdout review.
5. Streamlit connects to the private GitHub repository and executes `app.py`.
6. Streamlit uses Google Drive secrets to read/write the historical archive.

Never place Google service-account JSON, Dhan secrets or large historical
market-data archives in GitHub.
