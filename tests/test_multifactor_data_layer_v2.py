import pandas as pd
from multifactor_data_layer_v2 import FEATURES_59, attach_readiness_flags, readiness_report

def test_contract():
    assert len(FEATURES_59) == 59
    assert len(set(FEATURES_59)) == 59

def test_zero_is_known_state():
    df=pd.DataFrame({
        "symbol":["ABC"],
        "decision_timestamp":pd.to_datetime(["2026-01-02"],utc=True),
        "ret_1":[0.01],
        "market_ret_1":[0.005],
        "news_available":[0],
        "event_available":[1],
        "fno_available":[0],
        "fundamental_eps":[5.0],
    })
    x=attach_readiness_flags(df)
    assert int(x.loc[0,"ready_news"])==0
    assert pd.notna(x.loc[0,"ready_news"])
    assert 0 <= readiness_report(df)["mean_group_readiness_pct"] <= 100
