import argparse, sys, yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import femnist_partition
from src.models import FemnistCNN
from src.fl import run_federated

p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--output',default='outputs/femnist'); a=p.parse_args()
cfg=yaml.safe_load(open(a.config,'r',encoding='utf-8'))
clients,test=femnist_partition(cfg['data_root'],cfg['num_clients'])
for seed in cfg['seeds']:
    r=run_federated(FemnistCNN(),clients,test,cfg,seed,Path(a.output)/str(seed))
    print(seed,r)
