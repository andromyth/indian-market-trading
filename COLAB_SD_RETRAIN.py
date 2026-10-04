"""Run in Google Colab (GPU not required). Weekly, or when status_validate.json says retrain=true.

!pip -q install -r requirements.txt
from google.colab import drive; drive.mount('/content/drive')
%run COLAB_SD_RETRAIN.py
"""
import json
from pathlib import Path

import pandas as pd

from selfdriving.sd_cycle import train_stage
from selfdriving.sd_global_cues import attach_global_features
from selfdriving.sd_news_features import build_news_features, score_news_frame
from selfdriving.sd_fno_features import build_fno_regime_features  # use before the existing next-session shift

BASE = Path("/content/drive/MyDrive/Indian_Market_Research")
SD = BASE / "selfdriving"

# 1. Panel produced by the existing builder (see INTEGRATION.md: panel.to_parquet(...))
panel = pd.read_parquet(BASE / "models/all_nse_delivery_multifactor_candidate_2026/PANEL_2026.parquet")

# 2. Global cues (PIT as-of join on decision_timestamp)
gp = SD / "global_cues.parquet"
if gp.exists():
    panel = attach_global_features(panel, pd.read_parquet(gp))

# 3. News features from the forward-only archive (columns become NaN before archive_start)
nr = SD / "raw_news.parquet"
if nr.exists():
    raw = pd.read_parquet(nr)
    nf = build_news_features(score_news_frame(raw), panel[["symbol", "decision_timestamp"]],
                             archive_start=str(pd.to_datetime(raw["published_at"], utc=True).min().date()))
    panel = panel.drop(columns=[c for c in nf.columns if c.startswith(("news_", "mkt_news_")) and c in panel.columns]) \
                 .merge(nf, on=["symbol", "decision_timestamp"], how="left")

# 4. Train, compare with champion on identical folds, gate, register
out = train_stage(panel, SD / "models")
(SD / "status_train.json").write_text(json.dumps(out, indent=2, default=str))
print(json.dumps({k: out.get(k) for k in ("status", "gate", "thresholds", "groups")}, indent=2, default=str))
