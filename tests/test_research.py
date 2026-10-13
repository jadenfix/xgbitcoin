import tempfile
import unittest
import numpy as np
import pandas as pd
from xgbitcoin import features,backtest,dataset,run

class ResearchTests(unittest.TestCase):
    def test_features_are_prefix_causal(self):
        close=pd.Series(np.exp(np.arange(150)*.002))
        original=features(close)
        close.iloc[80:]=9
        pd.testing.assert_frame_equal(features(close).iloc[:81],original.iloc[:81])
    def test_fee_ledger_entry_switch_exit(self):
        net,turn=backtest([1,1,0,1],[0,0,0,0],10)
        np.testing.assert_array_equal(turn,[1,0,1,2])
        self.assertAlmostEqual(net.sum(),-.004)
    def test_input_rejects_duplicates_and_negative_prices(self):
        with tempfile.TemporaryDirectory() as tmp:
            f,_=dataset(rows=128);f.loc[1,'timestamp']=f.loc[0,'timestamp']
            path=tmp+'/data.csv';f.to_csv(path,index=False)
            with self.assertRaises(ValueError):dataset(path)
        with self.assertRaises(ValueError):backtest([1],[0],-1)
    def test_actual_train_predict_reload_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            metrics,pred=run(output=tmp,rows=192)
            self.assertTrue(metrics['model_roundtrip_verified'])
            self.assertLess(metrics['train_last_timestamp'],metrics['test_first_timestamp'])
            self.assertTrue(np.isfinite(pred.equity).all())
            self.assertEqual(metrics['purged_rows'],1)
            self.assertGreater(metrics['test_rows'],20)
