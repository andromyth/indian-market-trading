# Multi-factor V2 build plan

1. Build in Google Colab from the multifactor-readiness-v2 branch.
2. Read the ALL-NSE universe and minimum-history gate.
3. Attach only historical PIT optional sources that carry valid timestamps.
4. Generate the data-readiness report.
5. Train next-session, 3-day, 5-day and 7-day candidate models.
6. Validate contracts and chronological holdout results.
7. Only then prepare a separate production-integration change.
