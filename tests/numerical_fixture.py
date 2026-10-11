"""Offline fixture; run after installing the package."""
import numpy as np
from pooledscreenaudit.cli import kernel
from threadpoolctl import threadpool_limits

core=kernel()
rng=np.random.default_rng(3);r=np.arange(712,832);x=rng.normal(size=(120,4))
y=np.column_stack([2*x[:,0]+rng.normal(size=120)*.1,x[:,1]+rng.normal(size=120)*.2,np.ones(120)])
y[:7,1]=np.nan;groups=(r-r.min())//5
with threadpool_limits(limits=1):
    a,_=core.eta(x,y,groups);scaled=y.copy();scaled[:,0]*=100
    b,_=core.eta(x,scaled,groups)
assert a['n_complete']==113 and a['n_eligible']==2
assert abs(a['equal_feature_eta']-b['equal_feature_eta'])<1e-10
assert abs(a['raw_complete_eta']-b['raw_complete_eta'])>1e-6
for fold in core.folds(groups,5):assert not set(groups[fold])&set(groups[~fold])
d=core.directions(r,(r>=770).astype(float)[:,None])
assert d['rank']==2 and np.max(np.abs(d['shared'][:2]))<1e-10
print('PASS: units, missingness, constant exclusion, grouped splits and rank-one geometry')
