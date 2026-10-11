import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from pooledscreenaudit.cli import validate, main

class ValidationEdgeTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(dict(variant=[f'A{i}C' for i in range(150)],residue=np.arange(150,dtype=float),wildtype='A',mutant='C',condition='x',y=np.arange(150)/100,weight=1.,T_depth=np.arange(150,dtype=float)))
    def test_nan_response(self):
        d=self.frame();d.loc[0,'y']=np.nan
        with self.assertRaises(ValueError):validate(d,'screen')
    def test_infinite_response(self):
        d=self.frame();d.loc[0,'y']=np.inf
        with self.assertRaises(ValueError):validate(d,'screen')
    def test_zero_weight(self):
        d=self.frame();d.loc[0,'weight']=0
        with self.assertRaises(ValueError):validate(d,'screen')
    def test_constant_response(self):
        d=self.frame();d.y=1
        with self.assertRaises(ValueError):validate(d,'screen')
    def test_all_missing_controls(self):
        d=self.frame();d.T_depth=np.nan
        with self.assertRaises(ValueError):validate(d,'screen')
    def test_partial_missing_controls(self):
        d=self.frame();d.loc[0,'T_depth']=np.nan
        self.assertEqual(validate(d,'screen')[0],['T_depth'])
    def test_infinite_control(self):
        d=self.frame();d.loc[0,'T_depth']=np.inf
        with self.assertRaises(ValueError):validate(d,'screen')
    def test_non_numeric_control(self):
        d=self.frame();d['T_depth']=d.T_depth.astype(object);d.loc[0,'T_depth']='oops'
        with self.assertRaises(ValueError):validate(d,'screen')
    def test_missing_column(self):
        with self.assertRaises(ValueError):validate(self.frame().drop(columns='condition'),'screen')
    def test_invalid_residue(self):
        d=self.frame();d.loc[0,'residue']=np.inf
        with self.assertRaises(ValueError):validate(d,'screen')
    def test_inconsistent_variant(self):
        d=self.frame();d.loc[1,'variant']=d.loc[0,'variant'];d.loc[1,'condition']='z'
        with self.assertRaises(ValueError):validate(d,'screen')
    def test_duplicate_site(self):
        d=self.frame();d.loc[1,'residue']=0
        with self.assertRaises(ValueError):validate(d,'geometry')
    def test_eta_missing_target(self):
        with self.assertRaises(ValueError):validate(self.frame(),'eta')
    def test_eta_constant_targets(self):
        d=self.frame();d['R_x']=1
        with self.assertRaises(ValueError):validate(d,'eta')
    def test_eta_mixed_targets(self):
        d=self.frame();d['R_x']=1;d['R_y']=np.sin(d.residue)
        self.assertEqual(len(validate(d,'eta')[1]),2)
    def test_eta_too_few_blocks(self):
        d=self.frame().head(30);d['R_x']=np.sin(d.residue)
        with self.assertRaises(ValueError):validate(d,'eta')
    def test_existing_output_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/'important.txt';f.write_text('keep')
            with self.assertRaises(SystemExit) as exc:main(['example','--out',td])
            self.assertEqual(exc.exception.code,2)
            self.assertEqual(f.read_text(),'keep')

if __name__=='__main__':unittest.main()
