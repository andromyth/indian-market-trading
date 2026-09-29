from pathlib import Path

app = Path("app.py").read_text()
fetcher = Path("data_fetcher.py").read_text()
assert 'st.form_submit_button("Analyze"' in app
assert 'st.session_state["watchlist_revision"]' in app
assert 'load_history(symbol, period, interval, revision)' in app
assert 'data_diagnostics' in app
assert 'def _yahoo_history' in fetcher
assert 'def _yahoo_chart_history' in fetcher
assert 'requests.get(url, params=params' in fetcher
assert 'import yfinance' not in fetcher
assert 'yf.' not in fetcher
assert '_local_persistent_history' in fetcher
print("dashboard submission/fallback tests: PASS")
