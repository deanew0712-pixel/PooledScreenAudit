"""Reproducible inner-training direction sensitivity conditional on frozen inputs."""
from pathlib import Path
import argparse, inspect, json, hashlib, shutil, sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import run_paper_a_phase7_directional_attribution_v1 as old

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--input-dir',type=Path);ap.add_argument('--limit',type=int);args=ap.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.input_dir:
        inp=args.input_dir.resolve();frame=pd.read_csv(inp/'frozen_edges.csv');spec=json.loads((inp/'split_manifest.json').read_text());tcols=json.loads((inp/'technical_columns.json').read_text())
    else:
        frame,_,tcols=old.load_enriched();spec=[]
        for ch,r,ix,mask in old.selected_splits(frame):spec.append(dict(challenge=ch,repeat=int(r),test_index=list(map(int,ix)),cold_index=list(map(int,frame.index[np.asarray(mask)]))))
        inp=out/'inputs';inp.mkdir(exist_ok=True);frame.to_csv(inp/'frozen_edges.csv',index=False);(inp/'technical_columns.json').write_text(json.dumps(tcols));(inp/'split_manifest.json').write_text(json.dumps(spec))
    if args.limit:spec=spec[:args.limit]
    old.OUT=out;old.load_enriched=lambda:(frame,[],tcols)
    old.selected_splits=lambda f:((s['challenge'],s['repeat'],np.array(s['test_index']),f.index.isin(s['cold_index'])) for s in spec)
    records=[];counter=[0]
    def basis_for(tr):
        sites=tr[['residue']+tcols].drop_duplicates('residue').sort_values('residue')
        return old.direction_basis(sites.residue.to_numpy(float),sites[tcols].to_numpy(float)),sites.residue.to_numpy(float)
    def fast_predict(tr,x,xt,a):
        xd=np.column_stack([np.ones(len(x)),x]);td=np.column_stack([np.ones(len(xt)),xt]);w=np.sqrt(tr.weight.to_numpy(float));xw=xd*w[:,None]
        penalty=np.eye(xd.shape[1])*a;penalty[0,0]=0
        return td@np.linalg.solve(xw.T@xw+penalty,xw.T@(tr.y.to_numpy(float)*w))
    # Verify the numerically equivalent BLAS implementation against the historical solver.
    rng=np.random.default_rng(103);tr=pd.DataFrame({'y':rng.normal(size=60),'weight':rng.uniform(.1,2,size=60)});x=rng.normal(size=(60,12));xt=rng.normal(size=(10,12))
    assert np.max(np.abs(fast_predict(tr,x,xt,1)-old.stable_ridge_from_design(tr,x,xt,1)))<1e-10
    cache={}
    def tune(train,test,tc,dt,dtest,subset,include_t,original_p=False):
        key=tuple(train.index)
        if key not in cache:
            outer,res=basis_for(train);parts=[];call=len(cache)
            for j,held in enumerate(np.array_split(np.array(sorted(train.residue.unique())),3)):
                mask=~train.residue.isin(held).to_numpy();tr=train.loc[mask];va=train.loc[~mask];b,ir=basis_for(tr)
                assert not set(ir)&set(held)
                v=old.direction_values(ir,b);ov=old.direction_values(ir,outer)
                row=dict(outer_index=call,inner_fold=j,training_positions=ir.tolist(),validation_positions=held.tolist(),shared=b['shared'].tolist(),coefficients=b['coefficients'].tolist(),p_center=b['p_center'].tolist(),training_only_abs_correlation=[float(abs(np.corrcoef(v[:,k],ov[:,k])[0,1])) for k in range(3)])
                records.append(row);parts.append((tr,va,old.direction_values(tr.residue.to_numpy(float),b),old.direction_values(va.residue.to_numpy(float),b)))
            cache[key]=(parts,old.direction_values(train.residue.to_numpy(float),outer),old.direction_values(test.residue.to_numpy(float),outer))
        parts,od,ot=cache[key];losses={float(a):[] for a in old.ALPHAS}
        for tr,va,di,dv in parts:
            x,xv=old.build_design(tr,va,tc,di,dv,subset,include_t,original_p)
            for a in losses:losses[a].append(old.wmae(va.y.to_numpy(),fast_predict(tr,x,xv,a),va.weight.to_numpy()))
        a=min(losses,key=lambda a:(np.mean(losses[a]),a));x,xt=old.build_design(train,test,tc,od,ot,subset,include_t,original_p)
        counter[0]+=1
        return a,fast_predict(train,x,xt,a)
    old.tune_and_predict=tune
    # Reuse unchanged evaluation, exact Shapley and decision accounting. 'global' is
    # a temporary legacy accounting key; its predictive basis is outer-training fitted.
    source=inspect.getsource(old.evaluate).replace('bases = {"global": global_basis, "fold_fitted": fold_basis}','bases = {"global": fold_basis}')
    ns=old.__dict__;exec(source,ns)
    with threadpool_limits(limits=1):result=ns['evaluate']()
    for f in out.glob('*.csv'):
        if f.name.startswith('._'):continue
        d=pd.read_csv(f)
        if 'basis_mode' in d:d['basis_mode']=d.basis_mode.replace({'global':'fully_nested'});d.to_csv(f,index=False)
    (out/'inner_direction_fit_audit.json').write_text(json.dumps(records,indent=2))
    s=json.loads((out/'analysis_summary.json').read_text());s.update(timing='Exploratory fully nested direction sensitivity fixed 2026-09-10',basis_mode='fully_nested',upstream_scope='Conditional on fixed assay-level DMSO atlas, count-derived labels and weights',outer_splits=spec,model_fits=counter[0],inner_direction_fits=len(records))
    (out/'analysis_summary.json').write_text(json.dumps(s,indent=2));print(json.dumps({k:v for k,v in s.items() if k!='outer_splits'},indent=2))

if __name__=='__main__':main()
