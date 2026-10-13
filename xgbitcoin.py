"""Chronological Bitcoin-direction research with actual CPU XGBoost."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from xgboost import Booster, DMatrix, train as xgb_train
from report import write_report


def dataset(csv=None, rows=512, seed=7):
    if csv:
        path=Path(csv);frame=pd.read_csv(path,parse_dates=["timestamp"])
        source={"kind":"user_csv","sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    else:
        if not 128<=rows<=100000: raise ValueError("rows must be128..100000")
        rng=np.random.default_rng(seed)
        returns=rng.normal(0,.018,rows)+.004*np.sin(np.arange(rows)/24)
        frame=pd.DataFrame({"timestamp":pd.date_range("2024-01-01",periods=rows,freq="D",tz="UTC"),
                            "close":40000*np.exp(np.cumsum(returns))})
        source={"kind":"seeded_synthetic","seed":seed,"not_bitcoin_market_history":True}
    frame["timestamp"]=pd.to_datetime(frame.timestamp,utc=True,errors="raise")
    if len(frame)<128 or frame.timestamp.isna().any() or not frame.timestamp.is_monotonic_increasing or frame.timestamp.duplicated().any():
        raise ValueError("Require128+ unique chronological timestamps")
    if not np.isfinite(frame.close).all() or (frame.close<=0).any(): raise ValueError("Prices must be finite positive")
    return frame,source


def features(close):
    # All features at execution timestamp t are available by t-1.
    ret=pd.Series(close).pct_change()
    return pd.DataFrame({"return_1":ret.shift(1),"return_3":ret.rolling(3).mean().shift(1),
                         "momentum_10":pd.Series(close).pct_change(10).shift(1),
                         "volatility_20":ret.rolling(20).std().shift(1)})


def backtest(position, forward_return, fee_bps=10):
    if not np.isfinite(fee_bps) or not 0<=fee_bps<=1000: raise ValueError("fee_bps must be0..1000")
    position=np.asarray(position,float);returns=np.asarray(forward_return,float)
    if position.shape!=returns.shape or not np.isfinite(returns).all() or not np.isin(position,[0,1]).all():
        raise ValueError("Invalid long/cash positions or returns")
    # Charge entry, rebalance and final exit. One unit of exposure, no leverage.
    turnover=np.abs(np.diff(np.r_[0,position]))
    if len(position): turnover[-1]+=position[-1]
    net=position*returns-turnover*fee_bps/10000
    return net,turnover


def run(csv=None,output="artifacts",rows=512,seed=7,fee_bps=10):
    frame,source=dataset(csv,rows,seed);x=features(frame.close)
    forward=frame.close.shift(-1)/frame.close-1
    valid=x.notna().all(axis=1)&forward.notna()
    indices=np.flatnonzero(valid);cut=int(len(indices)*.6)
    train=indices[:cut-1];test=indices[cut:] # One-bar purge across the target boundary.
    if len(train)<40 or len(test)<20: raise ValueError("Insufficient train/test observations")
    labels=(forward>0).astype(int)
    if labels.iloc[train].nunique()!=2: raise ValueError("Training data must contain both direction classes")
    model=xgb_train({"objective":"binary:logistic","max_depth":3,"eta":.06,"tree_method":"hist",
                 "nthread":1,"seed":seed},DMatrix(x.iloc[train],label=labels.iloc[train]),num_boost_round=48)
    probability=model.predict(DMatrix(x.iloc[test]))
    destination=Path(output);destination.mkdir(parents=True,exist_ok=True)
    model.save_model(destination/"model.ubj")
    restored=Booster();restored.load_model(destination/"model.ubj")
    np.testing.assert_allclose(probability,restored.predict(DMatrix(x.iloc[test])),rtol=0,atol=0)
    positions=(probability>=.55).astype(int)
    net,turnover=backtest(positions,forward.iloc[test],fee_bps)
    baseline,_=backtest(np.ones(len(test)),forward.iloc[test],fee_bps)
    result=pd.DataFrame({"timestamp":frame.timestamp.iloc[test].astype(str),"close":frame.close.iloc[test],
                         "p_up":probability,"position":positions,"forward_return":forward.iloc[test],
                         "net_return":net,"equity":np.cumprod(1+net),"buy_hold_equity":np.cumprod(1+baseline)})
    report={"scope":"Chronological fixed-model research; no live execution or profit promise","source":source,
            "train_rows":len(train),"test_rows":len(test),"train_last_timestamp":str(frame.timestamp.iloc[train[-1]]),
            "test_first_timestamp":str(frame.timestamp.iloc[test[0]]),"purged_rows":1,
            "features_available":"All features lag one bar; targets are following-close returns",
            "test_accuracy":float(accuracy_score(labels.iloc[test],probability>=.5)),
            "test_log_loss":float(log_loss(labels.iloc[test],probability,labels=[0,1])),
            "strategy_total_return":float(result.equity.iloc[-1]-1),"buy_hold_total_return":float(result.buy_hold_equity.iloc[-1]-1),
            "fee_bps":fee_bps,"turnover_units":float(turnover.sum()),"model_roundtrip_verified":True,
            "execution_assumption":"Fills at timestamp-t close, signal known at t-1; fee includes modeled slippage. No borrow, intrabar or capacity model."}
    (destination/"metrics.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    frame.to_csv(destination/"dataset.csv",index=False);result.to_csv(destination/"predictions.csv",index=False)
    write_report(destination/"index.html",report,result)
    print(json.dumps(report,indent=2));return report,result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--csv");p.add_argument("--output",default="artifacts")
    p.add_argument("--rows",type=int,default=512);p.add_argument("--seed",type=int,default=7)
    p.add_argument("--fee-bps",type=float,default=10);a=p.parse_args();run(a.csv,a.output,a.rows,a.seed,a.fee_bps)
if __name__=="__main__":main()
