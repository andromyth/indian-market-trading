#!/usr/bin/env python3
"""Scheduled NSE-first EOD ingestion.

Downloads the official NSE CM-UDiFF Common Bhavcopy for each requested trading
 day, caches a normalized copy, and optionally mirrors it to Google Drive.
This is an ingestion job, not a model trainer.
"""
from __future__ import annotations
import argparse
from datetime import date, timedelta
from nse_data_layer import fetch_nse_eod_bhavcopy

def business_days(start: date, end: date):
    d=start
    while d<=end:
        if d.weekday()<5:
            yield d
        d += timedelta(days=1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--sync-drive', action='store_true')
    args=ap.parse_args()
    ok=err=0
    for d in business_days(date.fromisoformat(args.start), date.fromisoformat(args.end)):
        try:
            df=fetch_nse_eod_bhavcopy(d, sync_drive=args.sync_drive)
            print(f'{d}: OK rows={len(df):,}')
            ok+=1
        except Exception as e:
            print(f'{d}: ERROR {type(e).__name__}: {e}')
            err+=1
    raise SystemExit(1 if ok==0 else 0)

if __name__=='__main__': main()
