import unittest
import numpy as np
import pandas as pd
from pooledscreenaudit.cli import validate, kernel

class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(dict(variant=[f'A{i}C' for i in range(20)],residue=np.arange(712,732),
                         wildtype='A',mutant='C',condition='drug',y=np.arange(20)/10,
                         weight=1.,T_depth=np.arange(20)))
    def test_valid(self):
        self.assertEqual(validate(self.df,'screen')[0],['T_depth'])
    def test_no_controls(self):
        with self.assertRaises(ValueError):validate(self.df.drop(columns='T_depth'),'screen')
    def test_infinite_weight(self):
        self.df.loc[0,'weight']=np.inf
        with self.assertRaises(ValueError):validate(self.df,'screen')
    def test_negative_weight(self):
        self.df.loc[0,'weight']=-1
        with self.assertRaises(ValueError):validate(self.df,'screen')
    def test_duplicate(self):
        with self.assertRaises(ValueError):validate(pd.concat([self.df,self.df]),'screen')
    def test_inconsistent_controls(self):
        d=self.df.copy();d.condition='other';d.T_depth+=1
        with self.assertRaises(ValueError):validate(pd.concat([self.df,d]),'screen')
    def test_invalid_amino_acid(self):
        self.df.loc[0,'mutant']='X'
        with self.assertRaises(ValueError):validate(self.df,'screen')
    def test_insufficient_sites(self):
        with self.assertRaises(ValueError):validate(self.df.head(5).copy(),'screen')
    def test_rank_one_partition(self):
        r=np.arange(712,980);d=kernel().directions(r,(r>=846).astype(float)[:,None])
        self.assertLess(np.abs(d['shared'][:2]).max(),1e-10)

if __name__=='__main__':unittest.main()
