"""Fetch pinned SRC counts and prepare explicitly defined matrix inputs."""
from pathlib import Path
import argparse,hashlib,json,re,urllib.request
import numpy as np
import pandas as pd

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--out',type=Path,required=True)
OUT=ap.parse_args().out
expected={x['file']:x['sha256'] for x in json.loads((Path(__file__).parent/'expected_SRC_PROVENANCE.json').read_text())['files']}
RAW=OUT/'source_inputs'; RAW.mkdir(parents=True,exist_ok=True)
SHA='956034184cbce1c65d34770557661b9a5905b2a3'
FILES=['DAS_25um_counts.csv','DAS_100um_counts.csv','Inhibitor_1_counts.csv','Inhibitor_2_counts.csv','Inhibitor_4_counts.csv']
AA=dict(zip(['Ala','Arg','Asn','Asp','Cys','Gln','Glu','Gly','His','Ile','Leu','Lys','Met','Phe','Pro','Ser','Thr','Trp','Tyr','Val'],list('ARNDCQEGHILKMFPSTWYV')))
manifest=[]; all_edges=[]; all_control=[]
for file in FILES:
    url=f'https://raw.githubusercontent.com/eahler/2023_src_inhib/{SHA}/data/{file}'
    path=RAW/file
    if not path.exists():path.write_bytes(urllib.request.urlopen(url,timeout=60).read())
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=expected[file]:raise ValueError(f'Input hash mismatch: {file}')
    manifest.append(dict(file=file,url=url,sha256=digest))
    df=pd.read_csv(path)
    assert set(df.timepoint)=={'Input','A','B','C'},(file,df.timepoint.unique())
    assert set(df.rep)=={'Rep1','Rep2'}
    total=df.groupby(['rep','timepoint']).counts.sum()
    number=df.groupby(['rep','timepoint']).variant.nunique()
    parsed=df.variant.str.extract(r'^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$')
    df['wildtype']=parsed[0].map(AA);df['residue']=pd.to_numeric(parsed[1]);df['mutant']=parsed[2].map(AA)
    df=df.dropna(subset=['wildtype','residue','mutant']);df=df[df.wildtype!=df.mutant].copy();df.residue=df.residue.astype(int)
    df['frequency']=[(c+.5)/(total.loc[r,t]+.5*number.loc[r,t]) for r,t,c in zip(df.rep,df.timepoint,df.counts)]
    wide=df.pivot(index=['variant','residue','wildtype','mutant'],columns=['rep','timepoint'],values='frequency')
    for rep in ['Rep1','Rep2']:wide[(rep,'Y')]=np.log2(wide[(rep,'C')]/wide[(rep,'Input')])
    edges=wide[[('Rep1','Y'),('Rep2','Y')]].copy();edges.columns=['y_A','y_B'];edges=edges.dropna().reset_index();edges['condition']=file.removesuffix('_counts.csv');edges['y']=edges[['y_A','y_B']].mean(axis=1);edges['weight']=1.
    all_edges.append(edges)
    for rep in ['Rep1','Rep2']:
        inp=df[(df.rep==rep)&(df.timepoint=='Input')]
        for residue,g in inp.groupby('residue'):
            c=g.counts.to_numpy(float);p=(c+.5)/(c.sum()+.5*len(c))
            all_control.append(dict(condition=edges.condition.iloc[0],rep=rep,residue=residue,log_depth=np.log1p(c.sum()),log_mean=np.log1p(c.mean()),log_sd=np.log1p(c.std()),entropy=-(p*np.log(p)).sum(),detected=(c>0).mean(),categories=len(c)))
edges=pd.concat(all_edges,ignore_index=True)
control=pd.DataFrame(all_control);control.to_csv(OUT/'src_input_qc_by_condition_replicate.csv',index=False)
# Mean input summaries across the five screens: construction uses no post-selection count.
site=control.groupby(['residue','rep']).mean(numeric_only=True).unstack('rep')
site.columns=[f'T_{r}_{c}' for c,r in site.columns];site=site.reset_index()
edges=edges.merge(site,on='residue',validate='many_to_one')
edges.to_csv(OUT/'src_edges.csv',index=False)
(OUT/'SRC_PROVENANCE.json').write_text(json.dumps(dict(commit=SHA,files=manifest,edges=len(edges),variants=edges.variant.nunique(),positions=edges.residue.nunique(),conditions=edges.condition.unique().tolist(),response='mean replicate log2 endpoint C/input frequency ratio; not published activity score',control='site input-count summaries averaged across conditions; two replicates; not DMSO',license_note='Upstream repository has no explicit license; raw files are referenced and fetched, not relicensed or redistributed in the submission workflow.'),indent=2))
print((OUT/'SRC_PROVENANCE.json').read_text())
