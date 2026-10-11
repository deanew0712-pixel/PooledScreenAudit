"""Portable matrix-level sensitivity audit; see README for scope and input schema."""
from pathlib import Path
import argparse,itertools,json,math,platform,time
import numpy as np
import pandas as pd
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

ALPHAS=[.01,.1,1.,10.,100.]
AA=list('ACDEFGHIKLMNPQRSTVWY')

def prep(a):
    a=np.asarray(a,float)
    med=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in a.T])
    f=np.where(np.isfinite(a),a,med);center=f.mean(axis=0);scale=f.std(axis=0);scale[scale<1e-12]=1
    return med,center,scale

def transform(a,p):
    med,center,scale=p
    return (np.where(np.isfinite(a),a,med)-center)/scale

def ridge(x,y,xt,alpha,w=None):
    # Inputs have already been transformed using this fit's training observations.
    w=np.ones(len(x)) if w is None else np.asarray(w,float)
    w=w/w.mean();xm=np.average(x,axis=0,weights=w);ym=np.average(y,axis=0,weights=w)
    xc=x-xm;yc=y-ym
    gram=xc.T@(xc*w[:,None]);gram.flat[::len(gram)+1]+=alpha
    rhs=xc.T@(yc*w[:,None] if yc.ndim==2 else yc*w)
    beta=solve(gram,rhs,assume_a='pos',check_finite=False)
    return (xt-xm)@beta+ym

def folds(groups,k):
    levels=np.sort(np.unique(groups))
    return [np.isin(groups,levels[i::k]) for i in range(k)]

def eta(x,y,groups):
    pred=np.full_like(y,np.nan);base=pred.copy();scales=pred.copy();selected=[]
    for test in folds(groups,5):
        tr=~test;losses=[]
        for alpha in ALPHAS:
            ls=[]
            for val in folds(groups[tr],3):
                fit=~val;xp=prep(x[tr][fit]);yp=prep(y[tr][fit])
                yh=ridge(transform(x[tr][fit],xp),transform(y[tr][fit],yp),transform(x[tr][val],xp),alpha)
                actual=transform(y[tr][val],yp);mask=np.isfinite(y[tr][val]);ls.append(np.mean((actual[mask]-yh[mask])**2))
            losses.append(np.mean(ls))
        alpha=ALPHAS[int(np.argmin(losses))];selected.append(alpha)
        xp=prep(x[tr]);yp=prep(y[tr]);z=ridge(transform(x[tr],xp),transform(y[tr],yp),transform(x[test],xp),alpha)
        pred[test]=z*yp[2]+yp[1];base[test]=yp[1];scales[test]=yp[2]
    valid=np.isfinite(y)&np.isfinite(pred);complete=valid.all(axis=1)
    e=np.where(valid,(y-pred)**2,0);b=np.where(valid,(y-base)**2,0)
    se=(e/scales**2).sum(axis=0);sb=(b/scales**2).sum(axis=0)
    eligible=sb>1e-12;rs=np.full(y.shape[1],np.nan);rs[eligible]=1-se[eligible]/sb[eligible]
    equal=float(np.mean(rs[eligible]));raw=1-float(e.sum()/b.sum());cc=1-float(e[complete].sum()/b[complete].sum())
    rows=pd.DataFrame(dict(n_observed=valid.sum(axis=0),scaled_sse=se,scaled_sst=sb,r2=rs,raw_sse=e.sum(axis=0),raw_sst=b.sum(axis=0)))
    return dict(n_positions=len(y),n_complete=int(complete.sum()),n_features=y.shape[1],n_eligible=int(eligible.sum()),equal_feature_r2=equal,equal_feature_eta=math.sqrt(max(0,equal)),raw_pooled_r2=raw,raw_pooled_eta=math.sqrt(max(0,raw)),raw_complete_r2=cc,raw_complete_eta=math.sqrt(max(0,cc)),alphas=selected),rows

def pbasis(r):
    z=(np.asarray(r,float)-797)/100
    return np.column_stack([z,z*z,z*z*z])

def directions(r,t):
    p=pbasis(r);center=p.mean(axis=0);p=p-center
    c=np.column_stack([np.ones(len(r)),transform(t,prep(t))]);u,s,_=np.linalg.svd(c,full_matrices=False)
    rank=int((s>max(c.shape)*np.finfo(float).eps*s[0]).sum());q=u[:,:rank]
    a=p.T@(p-q@(q.T@p));b=p.T@p
    ev,evec=np.linalg.eigh(b);wh=evec@np.diag(1/np.sqrt(ev));core=wh.T@a@wh
    vals,vecs=np.linalg.eigh((core+core.T)/2);order=np.argsort(vals)[::-1];coeff=wh@vecs[:,order]
    return dict(center=center,coeff=coeff,shared=1-np.clip(vals[order],0,1),rank=rank)

def geometry(sites,tcols,out):
    r=sites.residue.to_numpy();rows=[];rng=np.random.default_rng(20260905)
    def add(label,group,rep=0):
        one=pd.get_dummies(group).to_numpy(float);d=directions(r,one)
        rows.append(dict(definition=label,replicate=rep,groups=len(np.unique(group)),centered_rank=d['rank']-1,**{f'shared_v{i+1}':v for i,v in enumerate(d['shared'])}))
    source=(r>=846).astype(int);bins=((r-r.min())//20).astype(int)
    add('source_two_subamplicons',source);add('analyst_20aa',bins)
    for cut in r[19:-20]:add('two_contiguous_cutpoint',r>=cut,int(cut))
    for i in range(200):
        add('random_two_same_sizes',rng.permutation(source),i);add('random_bins_same_sizes',rng.permutation(bins),i)
    d=directions(r,sites[tcols].to_numpy(float));rows.append(dict(definition='T',replicate=0,groups=0,centered_rank=d['rank']-1,**{f'shared_v{i+1}':v for i,v in enumerate(d['shared'])}))
    pd.DataFrame(rows).to_csv(out/'partition_geometry.csv',index=False)

def design(df,mods,tcols,db,conds):
    raw=np.column_stack([(df.wildtype==a).to_numpy(float) for a in AA]+[(df.mutant==a).to_numpy(float) for a in AA])
    if 'T' in mods:raw=np.column_stack([raw,df[tcols].to_numpy(float)])
    if 'P' in mods:raw=np.column_stack([raw,pbasis(df.residue)])
    inds=[int(v[1:])-1 for v in mods if v.startswith('v')]
    if inds:raw=np.column_stack([raw,((pbasis(df.residue)-db['center'])@db['coeff'])[:,inds]])
    cond=np.column_stack([(df.condition==c).to_numpy(float) for c in conds])
    return raw,cond

def fitted_design(raw,cond,p):
    z=transform(raw,p);return np.column_stack([cond,z,np.einsum('ij,ik->ijk',z,cond).reshape(len(z),-1)])

def predict(train,test,mods,tcols,db,conds,outcome):
    raw,cond=design(train,mods,tcols,db,conds);rt,ct=design(test,mods,tcols,db,conds)
    y=train[outcome].to_numpy(float);w=train.weight.to_numpy(float);loss=[]
    for alpha in ALPHAS:
        ls=[]
        for val in folds(train.residue.to_numpy(),3):
            p=prep(raw[~val]);pred=ridge(fitted_design(raw[~val],cond[~val],p),y[~val],fitted_design(raw[val],cond[val],p),alpha,w[~val]);ls.append(np.average(np.abs(y[val]-pred),weights=w[val]))
        loss.append(np.mean(ls))
    alpha=ALPHAS[int(np.argmin(loss))];p=prep(raw)
    return ridge(fitted_design(raw,cond,p),y,fitted_design(rt,ct,p),alpha,w),alpha

def screen(df,tcols,out,full=True):
    assert df.groupby('variant').residue.nunique().max()==1
    assert np.isfinite(df.y).all() and (df.weight>0).all()
    rng=np.random.default_rng(20260905);variants=np.sort(df.variant.unique());positions=np.sort(df.residue.unique());rng.shuffle(positions)
    splits=[]
    for i in range(5):
        chosen=np.random.default_rng(20260905+i).choice(variants,math.ceil(.2*len(variants)),replace=False);splits.append(('variant_cold',i,df.variant.isin(chosen).to_numpy()))
    for i,pp in enumerate(np.array_split(positions,5)):splits.append(('position_cold',i,df.residue.isin(pp).to_numpy()))
    subsets=[tuple(f'v{i+1}' for i in range(3) if mask>>i&1) for mask in range(8)]
    models={'baseline':(),'P':('P',),'T':('T',),'TP':('T','P')}
    if full:
        for sub in subsets:
            if sub:models['+'.join(sub)]=sub;models['T+'+'+'.join(sub)]=('T',)+sub
    rows=[];geo=[];condlist=sorted(df.condition.unique())
    for challenge,fold,testmask in splits:
        tr=df[~testmask];te=df[testmask];sites=tr[['residue']+tcols].drop_duplicates('residue').sort_values('residue');db=directions(sites.residue,sites[tcols].to_numpy(float))
        geo.append(dict(challenge=challenge,fold=fold,n_train_positions=len(sites),rank=db['rank'],**{f'shared_v{i+1}':v for i,v in enumerate(db['shared'])}))
        for name,mods in models.items():
            yp,alpha=predict(tr,te,mods,tcols,db,condlist,'y')
            for c in ['ALL']+condlist:
                mask=np.ones(len(te),bool) if c=='ALL' else (te.condition==c).to_numpy();part=te.loc[mask];tc=tr if c=='ALL' else tr[tr.condition==c];iqr=tc.y.quantile(.75)-tc.y.quantile(.25)
                rows.append(dict(challenge=challenge,fold=fold,model=name,condition=c,n_train=len(tr),n_test=len(part),n_test_variants=part.variant.nunique(),alpha=alpha,mae=np.average(np.abs(part.y.to_numpy()-yp[mask]),weights=part.weight),train_iqr=iqr))
        print(challenge,fold,'complete',flush=True)
    metrics=pd.DataFrame(rows);metrics.to_csv(out/'screen_metrics.csv',index=False);pd.DataFrame(geo).to_csv(out/'screen_direction_geometry.csv',index=False)
    gains=[]
    for key,g in metrics.groupby(['challenge','fold','condition']):
        g=g.set_index('model');b=g.loc['baseline','mae'];t=g.loc['T','mae'];iqr=g.loc['baseline','train_iqr']
        gains.append(dict(challenge=key[0],fold=key[1],condition=key[2],baseline_mae=b,T_mae=t,P_gain=b-g.loc['P','mae'],P_given_T=t-g.loc['TP','mae'],P_relative=(b-g.loc['P','mae'])/b,P_given_T_relative=(t-g.loc['TP','mae'])/t,P_iqr=(b-g.loc['P','mae'])/iqr))
    gain=pd.DataFrame(gains);gain.to_csv(out/'screen_gains.csv',index=False)
    if full:
        shp=[]
        for key,g in metrics[metrics.condition=='ALL'].groupby(['challenge','fold']):
            ma=g.set_index('model').mae.to_dict()
            for use_t in [False,True]:
                base=ma['T' if use_t else 'baseline']
                def v(sub):return 0. if not sub else base-ma[('T+' if use_t else '')+'+'.join(sub)]
                for i in range(3):
                    phi=0.;target=f'v{i+1}'
                    for sub in subsets:
                        if target in sub:continue
                        k=len(sub);phi+=math.factorial(k)*math.factorial(2-k)/6*(v(tuple(sorted(sub+(target,))))-v(sub))
                    shp.append(dict(challenge=key[0],fold=key[1],include_T=int(use_t),direction=target,shapley_gain=phi,full_gain=v(('v1','v2','v3'))))
        pd.DataFrame(shp).to_csv(out/'screen_shapley.csv',index=False)
    summary=gain.groupby(['challenge','condition']).mean(numeric_only=True).reset_index();summary.to_csv(out/'screen_summary.csv',index=False)
    return summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['eta','geometry','screen']);ap.add_argument('--input',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);ap.add_argument('--controls',default='T_');ap.add_argument('--targets',default='R_');ap.add_argument('--simple',action='store_true');args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);df=pd.read_csv(args.input);tc=[c for c in df if c.startswith(args.controls)];assert tc
    with threadpool_limits(limits=1):
        if args.command=='screen':screen(df,tc,args.out,not args.simple)
        elif args.command=='geometry':geometry(df,tc,args.out)
        else:
            rc=[c for c in df if c.startswith(args.targets)];assert rc
            x=np.column_stack([df[tc],pbasis(df.residue)]);y=df[rc].to_numpy(float);rows=[]
            for width in [15,20,30]:
                groups=((df.residue-df.residue.min())//width).to_numpy();res,features=eta(x,y,groups);res['width']=width;rows.append(res);features.insert(0,'feature',rc);features.to_csv(args.out/f'eta_features_{width}.csv',index=False)
            groups=((df.residue-df.residue.min())//20).to_numpy()
            for label,mult in [('unit_sasa_x_0_1',.1),('unit_sasa_x_10',10.)]:
                yy=y.copy();yy[:,['sasa' in c for c in rc]]*=mult;res,_=eta(x,yy,groups);res.update(width=20,sensitivity=label);rows.append(res)
            keep=np.array(['sasa' not in c for c in rc]);res,_=eta(x,y[:,keep],groups);res.update(width=20,sensitivity='without_sasa');rows.append(res)
            (args.out/'eta_summary.json').write_text(json.dumps(rows,indent=2))
    (args.out/'runtime.json').write_text(json.dumps(dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,command=vars(args)),default=str,indent=2))

if __name__=='__main__':main()
