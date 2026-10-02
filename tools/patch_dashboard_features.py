from pathlib import Path

APP = Path("app.py")
text = APP.read_text(encoding="utf-8")

IMPORT_MARKER = "from prediction_history_ui import render_delivery_forecast, render_prediction_history\n"
if IMPORT_MARKER not in text:
    anchor = "from historical_replay import (\n    historical_price, historical_benchmark, historical_news, historical_announcements,\n    actual_outcome, replay_rule_delivery, _entry_exit_metrics,\n)\n"
    if anchor not in text:
        raise SystemExit("historical_replay import anchor not found")
    text = text.replace(anchor, anchor + IMPORT_MARKER, 1)

old_view = 'analysis_view = st.sidebar.radio(\n    "View", ["Live Dashboard", "Historical Replay"], index=0,'
new_view = 'analysis_view = st.sidebar.radio(\n    "View", ["Live Dashboard", "Historical Replay", "Prediction History"], index=0,'
if old_view in text:
    text = text.replace(old_view, new_view, 1)
elif new_view not in text:
    raise SystemExit("View radio anchor not found")

branch_marker = 'if analysis_view == "Prediction History":\n'
if branch_marker not in text:
    anchor = 'style = st.sidebar.radio("Mode", ["Delivery", "Intraday"], index=0)\n'
    if anchor not in text:
        raise SystemExit("mode radio anchor not found")
    branch = '''if analysis_view == "Prediction History":\n    render_prediction_history()\n    st.stop()\n\n'''
    text = text.replace(anchor, branch + anchor, 1)

render_marker = '    render_delivery_forecast(direct_delivery_results)\n'
if render_marker not in text:
    anchor = 'for rec in live_records:\n'
    if anchor not in text:
        raise SystemExit("live record rendering anchor not found")
    block = '''if style == "Delivery" and direct_delivery_results:\n    render_delivery_forecast(direct_delivery_results)\n\n'''
    text = text.replace(anchor, block + anchor, 1)

APP.write_text(text, encoding="utf-8")
print("dashboard feature patch applied")

# Trigger the workflow once after it was installed; the patch itself is idempotent.
