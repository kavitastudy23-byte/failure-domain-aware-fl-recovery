import argparse, sys, yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import cifar10_partition
from src.models import CifarCNN
from src.fl import run_federated

p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--output',default='outputs/cifar10'); a=p.parse_args()
cfg=yaml.safe_load(open(a.config,'r',encoding='utf-8'))
for seed in cfg['seeds']:
    clients,test=cifar10_partition(cfg['data_root'],cfg['num_clients'],cfg['dirichlet_alpha'],seed)
    r=run_federated(CifarCNN(),clients,test,cfg,seed,Path(a.output)/str(seed))
    print(seed,r)
