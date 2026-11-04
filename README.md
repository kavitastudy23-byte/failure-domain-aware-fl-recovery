# Failure-Domain-Aware Adaptive Recovery Orchestration — Reproducibility Package

This package implements the experimental design described in the manuscript:

**Failure-Domain-Aware Adaptive Recovery Orchestration for Fault-Tolerant Federated Learning in Heterogeneous Edge Environments**

## Scientific scope

- CIFAR-10 and FEMNIST are public datasets.
- CIFAR-10 is partitioned across 20 clients using Dirichlet alpha=0.5; 10 clients participate per round; 50 rounds; SGD lr=0.01; up to 10 local minibatch updates per selected client per round.
- FEMNIST uses the LEAF writer partition with 100 registered clients; 10 selected per round; 30 rounds; SGD lr=0.01, momentum=0.9; approximately one local pass.
- Seeds: 101, 102, 103, 104, 105.
- Five logical failure domains.
- Controlled mechanism-level experiments cover independent failures, correlated-domain failures, recovery-policy comparison, ablation, and controller scalability.

The code does **not** force the paper's table values. It executes the described procedures and writes raw logs from which metrics are computed. Exact numerical agreement can depend on library versions, hardware, model initialization, and any execution details not preserved in the manuscript.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

# CIFAR-10 downloads automatically through torchvision.
python scripts/run_cifar.py --config configs/cifar10.yaml

# Prepare FEMNIST using the LEAF repository, then point data_root in configs/femnist.yaml to the generated JSON files.
python scripts/run_femnist.py --config configs/femnist.yaml

# Controlled recovery/domain experiments
python scripts/run_mechanism_experiments.py --config configs/mechanism.yaml

# Aggregate logs and create tables
python scripts/summarize_results.py --input outputs --out outputs/summary.csv
```

## FEMNIST dataset

The manuscript uses the **LEAF FEMNIST writer-based partition** rather than a random repartition. Obtain it from the official LEAF benchmark repository:

```bash
git clone https://github.com/TalwalkarLab/leaf.git
cd leaf/data/femnist
./preprocess.sh -s niid --sf 1.0 -k 0 -t sample
```

LEAF preprocessing options can change with repository versions. Preserve the generated `train/` and `test/` JSON files and record the exact LEAF commit hash in a final reproducibility release.

## Output layout

- `outputs/<dataset>/<seed>/metrics.json` — per-run metrics
- `outputs/mechanism/*.csv` — controlled failure/recovery event logs
- `outputs/summary.csv` — aggregated means

## Reference values from manuscript

`results_reference/` contains the values reported in the manuscript. These files are provided only as a comparison target and are **not read by the experiment code**.
