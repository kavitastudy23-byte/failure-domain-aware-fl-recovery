import argparse, json
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser(); p.add_argument('--input',default='outputs'); p.add_argument('--out',default='outputs/summary.csv'); a=p.parse_args()
records=[]
for f in Path(a.input).glob('*/*/metrics.json'):
    dataset=f.parents[1].name; seed=f.parent.name
    obj=json.load(open(f,'r',encoding='utf-8'))
    for method,m in obj.items(): records.append(dict(dataset=dataset,seed=seed,method=method,**m))
if records:
    df=pd.DataFrame(records); numeric=df.select_dtypes('number').columns
    s=df.groupby(['dataset','method'])[numeric].mean().reset_index(); s.to_csv(a.out,index=False); print(s.to_string(index=False))
for name in ['correlated_placement','recovery_policies','controller_scalability']:
    f=Path(a.input)/'mechanism'/f'{name}.csv'
    if f.exists():
        df=pd.read_csv(f); print('\n',name); print(df.groupby(df.columns[2] if name!='controller_scalability' else 'N').mean(numeric_only=True).to_string())
