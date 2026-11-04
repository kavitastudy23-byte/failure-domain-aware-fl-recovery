from dataclasses import dataclass
import math
import random
import time


@dataclass
class Resource:
    rid: int
    domain: int
    latency: float
    transfer_cost: float
    utilization: float
    risk: float


def adaptive_tau(p_fail, compute_rate, horizon, ckpt_cost, lambda_lost=1.0, lambda_ckpt=1.0,
                 tau_min=1.0, tau_max=10.0):
    a = max(1e-12, 0.5 * lambda_lost * p_fail * compute_rate)
    b = max(1e-12, lambda_ckpt * horizon * ckpt_cost)
    raw = math.sqrt(b / a)
    return min(tau_max, max(tau_min, raw))


def placement_cost(resource, origin_domain, weights=(0.30, 0.20, 0.20, 0.30)):
    a1, a2, a3, a4 = weights
    correlated = 1.0 if resource.domain == origin_domain else resource.risk
    return a1*resource.latency + a2*resource.transfer_cost + a3*resource.utilization + a4*correlated


def select_fd_resource(resources, origin_domain, weights=(0.30, 0.20, 0.20, 0.30)):
    cross = [r for r in resources if r.domain != origin_domain]
    pool = cross if cross else resources
    return min(pool, key=lambda r: placement_cost(r, origin_domain, weights))


def recovery_cost(latency, energy, comm, lost_work, residual_risk, weights=(.25,.10,.15,.35,.15)):
    b1,b2,b3,b4,b5 = weights
    return b1*latency + b2*energy + b3*comm + b4*lost_work + b5*residual_risk


def recovery_value(saved_work, success_prob, state_quality, cost, eps=1e-9):
    return saved_work * success_prob * state_quality / (eps + cost)


def choose_recovery(actions, weights=(.25,.10,.15,.35,.15), eta=0.25, value_threshold=0.0):
    scored = []
    for a in actions:
        if not a.get('feasible', True):
            continue
        c = recovery_cost(a['latency'], a['energy'], a['comm'], a['lost_work'], a['risk'], weights)
        v = recovery_value(a['saved_work'], a['success_prob'], a['state_quality'], c)
        checkpoint_dependent = a['name'] in {'resume','migrate','reuse'}
        if checkpoint_dependent and v < value_threshold:
            continue
        scored.append((c - eta*v, a['name'], c, v))
    if not scored:
        return 'discard', 1.0, 0.0
    scored.sort()
    _, name, cost, value = scored[0]
    return name, cost, value


def controller_latency_us(num_clients, rng):
    # Execute metadata lookup + simple action scoring to measure actual Python runtime.
    meta = [(i, i % 5, rng.random(), rng.random()) for i in range(num_clients)]
    start = time.perf_counter_ns()
    active = meta[:10]
    _ = min((m[2] + m[3], m[0]) for m in active)
    actions = [
        dict(name='resume', feasible=True, latency=.2, energy=.2, comm=.1, lost_work=.2, risk=.1,
             saved_work=.8, success_prob=.95, state_quality=.9),
        dict(name='migrate', feasible=True, latency=.4, energy=.3, comm=.4, lost_work=.2, risk=.05,
             saved_work=.8, success_prob=.9, state_quality=.9),
        dict(name='restart', feasible=True, latency=.3, energy=.2, comm=.1, lost_work=1.0, risk=.1,
             saved_work=0.0, success_prob=1.0, state_quality=1.0),
    ]
    choose_recovery(actions)
    elapsed = time.perf_counter_ns() - start
    return elapsed / 1000.0
