# xgbitcoin

A small, reproducible Bitcoin-direction research lab using actual CPU XGBoost, causal lagged features and a purged chronological holdout. Includes a standalone interactive report.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python xgbitcoin.py --rows 512 --output artifacts
python -m http.server 8000 --directory artifacts
```

Open http://localhost:8000 to explore the report. Default data are **seeded synthetic**, not historical BTC prices. Supply `--csv btc.csv` with `timestamp,close`,128+ unique chronological rows and finite positive closes for your own data. Dataset provenance, copied inputs, holdout predictions, model UBJ and metrics are saved together.

Features at t use closes through t−1. The first60% train the model with a one-row purge before the remaining chronological holdout; no random split or test-set tuning. A fixed0.55 probability threshold selects long/cash exposure. The cost ledger charges entries, changes and final exit; the fee defaults to10bps per unit turnover. A buy-and-hold baseline uses the same entry/exit cost convention. Fills are assumed at the close after a full bar of signal latency. This is a research simulation, with no liquidity/capacity modeling, exchange connection or profitability claim. Synthetic accuracy and returns demonstrate computation only.

The saved model is reloaded and its predictions must match before the report is written. Tests check prefix causality, input integrity, chronological splitting and explicit fee accounting.
