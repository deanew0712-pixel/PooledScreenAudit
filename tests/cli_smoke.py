"""Offline installed-wheel test; temporary directory can contain spaces."""
import hashlib, json, subprocess, sys, tempfile
from pathlib import Path
import pandas as pd

with tempfile.TemporaryDirectory(prefix='psa smoke ') as td:
    td=Path(td)
    def run(*args):
        subprocess.run([sys.executable,'-c','from pooledscreenaudit.cli import main; import sys; sys.exit(main())',*map(str,args)],check=True)
    run('example','--out',td/'input')
    run('geometry','--input',td/'input/egfr_sites.csv','--out',td/'geometry','--draws','5')
    run('eta','--input',td/'input/egfr_sites.csv','--out',td/'eta')
    g=pd.read_csv(td/'geometry/partition_geometry.csv')
    assert len(g)==7 and g.filter(like='shared_v').ge(-1e-10).all().all()
    for folder in ('geometry','eta'):
        for name in ('REPORT.md','RUN.json','OUTPUT_SHA256.json'):assert (td/folder/name).is_file()
    print('PASS installed CLI, paths with spaces, offline examples, geometry, eta and reports')
