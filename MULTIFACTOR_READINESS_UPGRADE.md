# Multi-factor readiness upgrade

The new candidate path tracks market price, market context, news, corporate
events, F&O and strict point-in-time fundamentals separately.

A source availability value of 0 is retained as a known negative/checked state;
it is not silently converted to missing data.

Model imputation is reported separately from source readiness.

The 3-stock certified intraday model and existing Delivery artifacts remain
untouched on this branch.
