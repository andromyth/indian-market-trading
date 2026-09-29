#!/usr/bin/env python3
"""Build a broader-NSE next-session Delivery candidate model in Colab.

This job is intentionally separate from Streamlit serving. It consumes the
cached official NSE CM-UDiFF daily bulk files produced by nse_bulk_eod_ingest.py,
builds the same 53-feature contract used by the current delivery artifact, and
walk-forward validates a pooled LogisticRegression/HGB pair across every symbol
for which sufficient history exists. Optional news/F&O/fundamental enrichment
can be added later when point-in-time datasets cover the wider universe.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error

FEATURES=[
'ret_1','ret_3','ret_6','ret_12','ret_24','ret_48','rsi14','atr14','ma20_ratio','ma50_ratio','volume_ratio','volume_z20','range_pct','body_pct','up_bar',
'market_ret_1','market_ret_3','market_ret_6','market_ret_12','market_rsi14','relative_ret_1','relative_ret_6','relative_ret_12','market_mood_proxy','market_stress_proxy',
'news_available','event_available','event_high_impact','fno_pcr','fno_oi_change','fno_iv_change','fno_available','fno_fut_close','fno_fut_return','fno_fut_oi',
'fundamental_revenue','fundamental_pat','fundamental_eps','fundamental_ebitda','fundamental_finance_cost','fundamental_total_equity','fundamental_total_debt','fundamental_cash',
'fund_revenue_growth','fund_pat_growth','fund_eps_growth','fund_ebitda_growth','fund_pat_margin','fund_debt_to_equity','fund_cash_to_debt','fundamental_age_days','fundamental_stale_90d','fundamental_stale_180d']


def load_bulk(root:Path)->pd.DataFrame:
    files=sorted(root.glob('**/BhavCopy_NSE_CM_0_0_0_*.parquet'))
    if not files: return pd.DataFrame()
    frames=[]
    for f in files:
        try:
            x=pd.read_parquet(f)
            frames.append(x)
        except Exception: pass
    if not frames: return pd.DataFrame()
    x=pd.concat(frames, ignore_index=True)
    x['symbol']=x['symbol'].astype(str).str.upper().str.strip()
    x['date']=pd.to_datetime(x['date'], errors='coerce').dt.normalize()
    for c in ['open','high','low','close','volume']:
        x[c]=pd.to_numeric(x[c],errors='coerce') if c in x else np.nan
    x=x.dropna(subset=['symbol','date','close','open','high','low'])
    return x.sort_values(['symbol','date']).drop_duplicates(['symbol','date'],keep='last')


def rsi(s,n=14):
    d=s.diff(); g=d.clip(lower=0).rolling(n).mean(); l=(-d.clip(upper=0)).rolling(n).mean(); rs=g/l
    out=100-100/(1+rs); out[l==0]=100; return out

def atr(x,n=14):
    c=x['close']; p=c.shift(1); tr=pd.concat([(x['high']-x['low']),(x['high']-p).abs(),(x['low']-p).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def feature_row(x, market):
    c=x['close']; h=x['high']; l=x['low']; o=x['open']; v=x['volume']
    last=c.iloc[-1]; ma20=c.rolling(20).mean().iloc[-1]; ma50=c.rolling(50).mean().iloc[-1]; vma=v.rolling(20).mean().iloc[-1]; vs=v.rolling(20).std().iloc[-1]
    mc=market['close'] if not market.empty else pd.Series(dtype=float)
    def ret(s,n):
        return float(s.iloc[-1]/s.iloc[-1-n]-1) if len(s)>n else np.nan
    row={
      'ret_1':ret(c,1),'ret_3':ret(c,3),'ret_6':ret(c,6),'ret_12':ret(c,12),'ret_24':ret(c,24),'ret_48':ret(c,48),
      'rsi14':float(rsi(c).iloc[-1]) if len(c)>15 else np.nan,'atr14':float(atr(x).iloc[-1]) if len(x)>15 else np.nan,
      'ma20_ratio':last/ma20-1 if pd.notna(ma20) and ma20 else np.nan,'ma50_ratio':last/ma50-1 if pd.notna(ma50) and ma50 else np.nan,
      'volume_ratio':float(v.iloc[-1]/vma) if pd.notna(vma) and vma else np.nan,
      'volume_z20':float((v.iloc[-1]-vma)/vs) if pd.notna(vma) and pd.notna(vs) and vs else np.nan,
      'range_pct':float((h.iloc[-1]-l.iloc[-1])/last) if last else np.nan,
      'body_pct':float((last-o.iloc[-1])/o.iloc[-1]) if o.iloc[-1] else np.nan,
      'up_bar':1.0 if last>o.iloc[-1] else 0.0,
      'market_ret_1':ret(mc,1),'market_ret_3':ret(mc,3),'market_ret_6':ret(mc,6),'market_ret_12':ret(mc,12),
      'market_rsi14':float(rsi(mc).iloc[-1]) if len(mc)>15 else np.nan,
    }
    for n in [1,6,12]: row[f'relative_ret_{n}']=row[f'ret_{n}']-row[f'market_ret_{n}'] if pd.notna(row[f'market_ret_{n}']) else np.nan
    row['market_mood_proxy']=row['market_ret_1']; row['market_stress_proxy']=float(mc.pct_change().rolling(20).std().iloc[-1]) if len(mc)>20 else np.nan
    for c in FEATURES:
        row.setdefault(c,np.nan)
    return row


def build_panel(x, min_history=60):
    rows=[]
    for sym,g in x.groupby('symbol',sort=False):
        g=g.sort_values('date').reset_index(drop=True)
        if len(g)<min_history+2: continue
        for i in range(min_history, len(g)-1):
            hist=g.iloc[:i+1]
            r=feature_row(hist, pd.DataFrame())
            r['symbol']=sym; r['date']=g.loc[i,'date']; r['y_return']=g.loc[i+1,'close']/g.loc[i,'close']-1; r['y_up']=1 if r['y_return']>0 else 0
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--bulk-root',default='/content/drive/MyDrive/Indian_Market_Research/processed/universe_expansion_2026_v4/bulk_eod'); ap.add_argument('--out',default='/content/drive/MyDrive/Indian_Market_Research/models/all_nse_delivery_candidate_2026'); args=ap.parse_args()
    root=Path(args.bulk_root); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    x=load_bulk(root)
    if x.empty: raise SystemExit('No official NSE bulk EOD parquet files found. Run nse_bulk_eod_ingest.py backfill first.')
    # Exclude clearly non-stock series already removed by the bulk normalizer.
    panel=build_panel(x)
    if panel.empty: raise SystemExit('No symbols have sufficient history for candidate training.')
    dates=sorted(panel['date'].dropna().unique()); cutoff=dates[max(1,int(len(dates)*0.8)-1)]
    tr=panel[panel.date<=cutoff]; te=panel[panel.date>cutoff]
    Xtr=tr[FEATURES].apply(pd.to_numeric,errors='coerce'); Xte=te[FEATURES].apply(pd.to_numeric,errors='coerce')
    ytr=tr.y_up.astype(int); yte=te.y_up.astype(int)
    clf=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler(with_mean=False)),('model',LogisticRegression(C=.5,max_iter=500,class_weight='balanced',random_state=42))])
    reg=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True)),('model',HistGradientBoostingRegressor(max_iter=120,learning_rate=.05,max_leaf_nodes=15,l2_regularization=1.5,random_state=42))])
    clf.fit(Xtr,ytr); reg.fit(Xtr,tr.y_return.astype(float))
    p=clf.predict_proba(Xte)[:,1] if len(te) else np.array([]); pr=reg.predict(Xte) if len(te) else np.array([])
    manifest={'engine_version':'2026-all-nse-delivery-candidate-1','model_family':'logistic_regression + hgb_return_regressor','feature_count':len(FEATURES),'features':FEATURES,'symbols':sorted(panel.symbol.unique().tolist()),'training_rows':int(len(tr)),'holdout_rows':int(len(te)),'cutoff':str(cutoff),'source':'NSE_OFFICIAL_UDIFF_BHAVCOPY','optional_information_policy':'News/F&O/fundamentals remain explicitly missing unless separate PIT datasets are merged; no current/future values are substituted.','metrics':{'holdout_accuracy':float(accuracy_score(yte,(p>=.5).astype(int))) if len(te) else None,'holdout_auc':float(roc_auc_score(yte,p)) if len(te) and yte.nunique()>1 else None,'holdout_mae_return':float(mean_absolute_error(te.y_return,pr)) if len(te) else None}}
    joblib.dump(clf,out/'deployment_all_nse_delivery_classifier_2026.joblib'); joblib.dump(reg,out/'deployment_all_nse_delivery_regressor_2026.joblib'); (out/'ALL_NSE_DELIVERY_MODEL_MANIFEST_2026.json').write_text(json.dumps(manifest,indent=2,default=str))
    print(json.dumps({k:manifest[k] for k in ['feature_count','symbols','training_rows','holdout_rows','metrics']},indent=2))

if __name__=='__main__': main()
