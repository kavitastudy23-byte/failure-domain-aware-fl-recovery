import copy
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from .recovery import adaptive_tau


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def train_local(model, dataset, device, lr, momentum, batch_size, local_steps=None, local_epochs=1):
    model = copy.deepcopy(model).to(device)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    done = 0
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = torch.nn.functional.cross_entropy(model(x), y); loss.backward(); opt.step()
            done += 1
            if local_steps is not None and done >= local_steps:
                return model.state_dict(), done
    return model.state_dict(), done


def fedavg(global_model, states, weights):
    out = copy.deepcopy(global_model.state_dict())
    total = float(sum(weights))
    for k in out:
        out[k] = sum(states[i][k].to(out[k].device) * (weights[i] / total) for i in range(len(states)))
    global_model.load_state_dict(out)


def evaluate(model, dataset, device, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval(); correct = 0; total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1); correct += (pred == y).sum().item(); total += y.numel()
    return 100.0 * correct / max(total, 1)


def run_federated(model, clients, testset, cfg, seed, outdir):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = random.Random(seed)
    p_fail = float(cfg['failure_probability'])
    K = int(cfg['clients_per_round'])
    rounds = int(cfg['rounds'])
    ck = cfg['checkpoint']
    method_names = ['fedavg_drop','restart','fixed_checkpoint','optimized_checkpoint','proposed']
    results = {}

    # Run each method with identical deterministic selection/failure traces per seed.
    client_trace = []
    failure_trace = []
    trace_rng = random.Random(seed + 999)
    for t in range(rounds):
        selected = trace_rng.sample(range(len(clients)), K)
        client_trace.append(selected)
        failure_trace.append({cid: (trace_rng.random() < p_fail) for cid in selected})

    for method in method_names:
        g = copy.deepcopy(model).to(device)
        recovered = failures = lost_steps = completed_steps = ckpt_bytes = 0
        for t in range(rounds):
            local_states, local_weights = [], []
            for cid in client_trace[t]:
                ds = clients[cid]
                failed = failure_trace[t][cid]
                failures += int(failed)
                local_steps = cfg.get('local_steps')
                local_epochs = cfg.get('local_epochs', 1)
                nominal_steps = int(local_steps or max(1, np.ceil(len(ds)/cfg['batch_size'])))
                completed_steps += nominal_steps

                if failed and method == 'fedavg_drop':
                    lost_steps += nominal_steps
                    continue

                if failed and method == 'restart':
                    # restart repeats prior work but returns a valid local update
                    lost_steps += nominal_steps
                    recovered += 1

                if method in {'fixed_checkpoint','optimized_checkpoint','proposed'}:
                    if method == 'fixed_checkpoint':
                        tau = max(1, nominal_steps // 2)
                    else:
                        tau = adaptive_tau(p_fail, 1.0, nominal_steps, ck['checkpoint_cost'],
                                           ck['lambda_lost'], ck['lambda_ckpt'], ck['tau_min'], ck['tau_max'])
                    # latest checkpoint determines exposed local work at failure
                    if failed:
                        exposed = min(nominal_steps, max(0, int(round(tau/2.0))))
                        lost_steps += exposed
                        recovered += 1
                    # approximate serialized model-state traffic using state tensors
                    state_bytes = sum(p.numel()*p.element_size() for p in g.state_dict().values())
                    n_ckpt = max(1, int(np.ceil(nominal_steps/max(tau,1e-9))))
                    ckpt_bytes += state_bytes * n_ckpt

                state, _ = train_local(g, ds, device, cfg['learning_rate'], cfg.get('momentum',0.0),
                                       cfg['batch_size'], local_steps, local_epochs)
                local_states.append(state); local_weights.append(len(ds))
            if local_states:
                fedavg(g, local_states, local_weights)
        accuracy = evaluate(g, testset, device)
        rsr = 100.0*recovered/failures if failures else 100.0
        lwr = 100.0*lost_steps/completed_steps if completed_steps else 0.0
        results[method] = dict(accuracy=accuracy, recovery_success_rate=rsr, lost_work_ratio=lwr,
                               useful_work_saved=max(0.0,100.0-lwr), checkpoint_bytes=ckpt_bytes,
                               failures=failures)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir/'metrics.json','w',encoding='utf-8') as f: json.dump(results,f,indent=2)
    return results
