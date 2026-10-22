"""Optimisers for the bonded parameters.

* **IBI** (moment matching): K <- K (sigma_CG^2/sigma_AA^2)^alpha, x0 <- x0 + alpha (<x>_AA - <x>_CG).
  Cheap (one simulation per iteration) and usually removes most of the systematic error of
  the Boltzmann-inversion guess (non-bonded forces between bonded beads, Jacobian).
* **PSO** in Swarm-CG style cycles: a particle swarm searches a box around the current best
  parameters; every PSO step evaluates the whole swarm in parallel. Force constants are
  searched on a logarithmic scale, equilibrium values linearly.

Both work on an ``evaluate(list[BondedParams]) -> list[EvalResult]`` callable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .params import BondedParams


CYCLE_WHAT = {"all": "bonds + angles", "bonds": "bonds only", "angles": "angles only"}


@dataclass
class CycleSpec:
    what: str = "all"            # "all" | "bonds" | "angles"
    k_factor: float = 3.0        # K searched in [K/f, K*f]
    r0_frac: float = 0.2         # r0 searched in r0 * (1 -/+ frac)
    theta_deg: float = 20.0      # theta0 searched in theta0 -/+ delta

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CYCLES = [CycleSpec("all", 3.0, 0.20, 20.0),
                  CycleSpec("angles", 2.0, 0.10, 10.0),
                  CycleSpec("all", 1.4, 0.06, 5.0)]


@dataclass
class OptimizerSettings:
    method: str = "ibi+pso"          # "ibi" | "pso" | "ibi+pso"
    ibi_iterations: int = 10         # maximum; stops earlier when the score levels off
    ibi_alpha: float = 0.7
    cycles: list = field(default_factory=lambda: [CycleSpec(**c.to_dict()) for c in DEFAULT_CYCLES])
    swarm_size: int = 0              # 0: 10 + 2 sqrt(D) (FST-PSO / Swarm-CG default)
    max_steps: int = 0               # 0: 8 + sqrt(D)
    patience: int = 4                # PSO steps / IBI iterations without improvement before stopping
    auto_stop: bool = True           # stop a stage once the score improves less than ``tolerance``
    tolerance: float = 0.01          # relative improvement that still counts (0.01 = 1 %)
    seed: int = 12345
    k_min: float = 0.1
    k_max_bond: float = 5000.0
    k_max_angle: float = 500.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cycles"] = [c.to_dict() if isinstance(c, CycleSpec) else dict(c) for c in self.cycles]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "OptimizerSettings":
        s = cls()
        for k, v in d.items():
            if k == "cycles":
                s.cycles = [CycleSpec(**c) for c in v]
            elif hasattr(s, k):
                setattr(s, k, v)
        return s


# Ready-made settings. Every stage stops by itself once the best score improves by less
# than ``tolerance`` for ``patience`` steps (as the early stopping of Swarm-CG); the numbers
# below are only upper limits.
PRESETS = {
    "quick": dict(method="ibi+pso", ibi_iterations=6, patience=3, tolerance=0.02,
                  cycles=[CycleSpec("all", 2.0, 0.10, 10.0)]),
    "standard": dict(method="ibi+pso", ibi_iterations=10, patience=4, tolerance=0.01,
                     cycles=[CycleSpec(**c.to_dict()) for c in DEFAULT_CYCLES]),
    "thorough": dict(method="ibi+pso", ibi_iterations=15, patience=6, tolerance=0.005,
                     cycles=[CycleSpec("all", 4.0, 0.25, 25.0)] + [CycleSpec(**c.to_dict()) for c in DEFAULT_CYCLES]),
}
PRESET_HELP = {
    "quick": "moment matching + one PSO cycle; a first result in a short time",
    "standard": "moment matching + three Swarm-CG cycles (wide search, angles, fine tuning)",
    "thorough": "longer patience, an extra wide cycle, stops only below 0.5 % improvement",
}


def preset(name: str, base: OptimizerSettings | None = None) -> OptimizerSettings:
    s = OptimizerSettings.from_dict(base.to_dict()) if base else OptimizerSettings()
    for k, v in PRESETS[name].items():
        setattr(s, k, [CycleSpec(**c.to_dict()) for c in v] if k == "cycles" else v)
    s.swarm_size = 0
    s.max_steps = 0
    s.auto_stop = True
    return s


def improved(new: float, best: float, tol: float) -> bool:
    """``new`` beats ``best`` by more than the relative tolerance."""
    if not np.isfinite(best):
        return np.isfinite(new)
    return best - new > tol * abs(best)


@dataclass
class EvalResult:
    index: int
    params: BondedParams
    score: float
    terms: dict = field(default_factory=dict)
    ok: bool = True
    message: str = ""
    stage: str = ""
    dists: dict | None = None        # CG distributions (thinned), for plots
    series: object = None            # TimeSeries of Rg and SASA of the CG run


class Tracker:
    """Keeps the best result and notifies the caller of every new one."""

    def __init__(self, on_result=None, on_best=None):
        self.best: EvalResult | None = None
        self.history: list[EvalResult] = []
        self.on_result = on_result
        self.on_best = on_best

    def add(self, results: list[EvalResult]) -> None:
        for r in results:
            self.history.append(r)
            if self.on_result:
                self.on_result(r)
            if r.ok and np.isfinite(r.score) and (self.best is None or r.score < self.best.score):
                self.best = r
                if self.on_best:
                    self.on_best(r)


# ----------------------------------------------------------------------- IBI
def run_ibi(start: BondedParams, evaluate, update, st: OptimizerSettings, tracker: Tracker,
            log=print, should_stop=None) -> BondedParams:
    """``update(params, cg_dists) -> params`` performs one moment-matching step.

    Stops after ``ibi_iterations`` updates, or earlier (``auto_stop``) when the best score
    has not improved by more than ``tolerance`` for ``patience`` iterations.
    """
    p = start
    last = None
    best = float("inf")
    stall = 0
    for it in range(st.ibi_iterations + 1):
        if should_stop and should_stop():
            break
        if last is not None:
            if not last.ok or last.dists is None:
                log("  IBI stopped: last simulation failed")
                break
            p = update(last.params, last.dists)
            p = clamp(p, st)
        res = evaluate([p], f"ibi {it}")[0]
        tracker.add([res])
        log(f"  IBI {it}: score {res.score:.4f}" + ("" if res.ok else f"  FAILED: {res.message}"))
        last = res
        score = res.score if res.ok else float("inf")
        if improved(score, best, st.tolerance):
            stall = 0
        else:
            stall += 1
        best = min(best, score)
        if st.auto_stop and stall >= min(st.patience, 2):
            log(f"  IBI converged: less than {100 * st.tolerance:g} % improvement for {stall} iterations")
            break
    return tracker.best.params if tracker.best else start


def clamp(p: BondedParams, st: OptimizerSettings) -> BondedParams:
    q = p.copy()
    for g, (K, r0) in q.bonds.items():
        q.bonds[g] = (float(np.clip(K, st.k_min, st.k_max_bond)), float(max(r0, 0.01)))
    for g, (K, th) in q.angles.items():
        q.angles[g] = (float(np.clip(K, st.k_min, st.k_max_angle)), float(np.clip(th, 0.0, 180.0)))
    return q


# ----------------------------------------------------------------------- PSO
W_INERTIA, C_COG, C_SOC = 0.729, 1.494, 1.494


def _space(p: BondedParams, spec: CycleSpec, st: OptimizerSettings):
    """Free parameter keys and their (centre, lower, upper) in search coordinates."""
    keys, lo, hi, x0, logk = [], [], [], [], []
    for key in p.keys():
        kind, _g, i = key
        if spec.what == "bonds" and kind != "bond":
            continue
        if spec.what == "angles" and kind != "angle":
            continue
        v = p.get(key)
        if i == 0:
            kmax = st.k_max_bond if kind == "bond" else st.k_max_angle
            v = float(np.clip(v, st.k_min, kmax))
            c = np.log(v)
            a = np.log(max(st.k_min, v / spec.k_factor))
            b = np.log(min(kmax, v * spec.k_factor))
            logk.append(True)
        elif kind == "bond":
            c, a, b = v, v * (1 - spec.r0_frac), v * (1 + spec.r0_frac)
            logk.append(False)
        else:
            c, a, b = v, max(0.0, v - spec.theta_deg), min(180.0, v + spec.theta_deg)
            logk.append(False)
        if b - a < 1e-9:
            b = a + 1e-6
        keys.append(key)
        x0.append(float(np.clip(c, a, b)))
        lo.append(a)
        hi.append(b)
    return keys, np.array(x0), np.array(lo), np.array(hi), np.array(logk, bool)


def _to_params(base: BondedParams, keys, x, logk) -> BondedParams:
    q = base.copy()
    for key, v, lg in zip(keys, x, logk):
        q.set(key, float(np.exp(v)) if lg else float(v))
    return q


def run_pso(start: BondedParams, evaluate, st: OptimizerSettings, tracker: Tracker, n_jobs: int = 1,
            log=print, should_stop=None, on_cycle=None) -> BondedParams:
    rng = np.random.default_rng(st.seed)
    best = start
    for ci, spec in enumerate(st.cycles, 1):
        keys, x0, lo, hi, logk = _space(best, spec, st)
        D = len(keys)
        if D == 0:
            log(f"PSO cycle {ci}: nothing to optimise ({spec.what})")
            continue
        n = st.swarm_size or int(10 + 2 * np.sqrt(D))
        if not st.swarm_size and n_jobs > 1:
            n = int(np.ceil(n / n_jobs) * n_jobs)      # no idle cores in a PSO step
        n = max(n, 2)
        steps = st.max_steps or int(8 + np.sqrt(D))
        log(f"PSO cycle {ci}/{len(st.cycles)} ({spec.what}): {D} parameters, {n} particles, "
            f"up to {steps} steps = {n * steps} simulations")
        if on_cycle:
            on_cycle(ci, len(st.cycles), n * steps)
        span = hi - lo
        X = lo + rng.uniform(size=(n, D)) * span
        X[0] = x0
        V = rng.uniform(-1, 1, size=(n, D)) * 0.2 * span
        vmax = 0.5 * span
        pbest = X.copy()
        pscore = np.full(n, np.inf)
        gbest, gscore = x0.copy(), np.inf
        # the centre has usually been evaluated already
        if tracker.best is not None and _same(tracker.best.params, _to_params(best, keys, x0, logk)):
            gscore = tracker.best.score
        stall = 0
        for step in range(steps):
            if should_stop and should_stop():
                return tracker.best.params if tracker.best else best
            plist = [_to_params(best, keys, x, logk) for x in X]
            res = evaluate(plist, f"pso c{ci} s{step + 1}")
            tracker.add(res)
            sc = np.array([r.score if r.ok else np.inf for r in res])
            better = sc < pscore
            pbest[better] = X[better]
            pscore[better] = sc[better]
            gain = improved(float(np.min(sc)), gscore, st.tolerance)
            if np.min(sc) < gscore:
                gscore = float(np.min(sc))
                gbest = X[int(np.argmin(sc))].copy()
            stall = 0 if gain else stall + 1
            log(f"  step {step + 1}/{steps}: best {np.min(sc):.4f}, mean {np.mean(sc[np.isfinite(sc)]) if np.isfinite(sc).any() else float('inf'):.4f}, "
                f"cycle best {gscore:.4f}")
            if st.auto_stop and stall >= st.patience:
                log(f"  converged: less than {100 * st.tolerance:g} % improvement for {st.patience} steps, "
                    f"cycle {ci} done")
                break
            r1 = rng.uniform(size=(n, D))
            r2 = rng.uniform(size=(n, D))
            V = W_INERTIA * V + C_COG * r1 * (pbest - X) + C_SOC * r2 * (gbest - X)
            V = np.clip(V, -vmax, vmax)
            X = X + V
            out = (X < lo) | (X > hi)
            V[out] *= -0.5                      # damped reflection at the walls
            X = np.clip(X, lo, hi)
        if np.isfinite(gscore):
            best = _to_params(best, keys, gbest, logk)
        log(f"PSO cycle {ci} finished: best score {gscore:.4f}")
    return tracker.best.params if tracker.best else best


def _same(a: BondedParams, b: BondedParams, tol: float = 1e-9) -> bool:
    ka, kb = a.keys(), b.keys()
    return ka == kb and all(abs(a.get(k) - b.get(k)) <= tol * max(1.0, abs(a.get(k))) for k in ka)
