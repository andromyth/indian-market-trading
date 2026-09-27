# Model build and release workflow

## 1. Data collection

Use the Streamlit **NSE Data Layer** controls or a local/Colab acquisition job
to pull NSE official data and persist it to Google Drive.

Do not put historical market data into GitHub.

## 2. Build in Google Colab

Mount Google Drive and run:

```bash
python colab_build_nse_model.py \
  --drive-root /content/drive/MyDrive/Indian_Market_Research \
  --symbols RELIANCE BHARTIARTL HDFCBANK TCS INFY
```

The script creates a chronological holdout and explicitly validates:

- class 1 is UP;
- class 0 is DOWN/FLAT;
- `predict_proba()[:,1]` is P(UP);
- future data are not used as predictors;
- the serialized model reloads successfully.

## 3. Promote only after validation

The Colab output is a **candidate**, not an automatic production model.
Promote it only after review of the holdout metrics, feature coverage,
calibration and historical replay results.

Copy the small release artifacts into the private GitHub repository's
`production_model/` directory. Keep Google Drive data out of the repository.

## 4. Streamlit

Streamlit loads the model artifact at runtime and reads current/historical data
from Google Drive. It must never retrain the model.
