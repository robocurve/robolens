# RoboLens — Foundation Design

> **Goal:** RoboLens is the "Inspect AI for robotics" — an open-source evaluation
> framework for **physical AI / VLA (Vision-Language-Action) models**. It lets
> anyone define a robotics benchmark once and run *any* VLA policy against *any*
> compatible embodiment (real robot or simulator), with first-class logging to
> [Rerun](https://github.com/rerun-io/rerun) and structured eval logs.
>
> This repo is the *framework* ("Inspect AI"). Concrete benchmarks
> ("Inspect Evals for robotics") are explicitly **out of scope** and live in a
> separate repo. We ship only the minimal reference evals needed to exercise and
> document the framework.

## 0. Design provenance & key insight

Inspect AI's core decomposition is `Task = dataset + solver + scorer`, with a
pluggable `Model` and an `eval()` entry point that produces an immutable
`EvalLog`. The genius is the clean separation between *what you evaluate*
(`Task`) and *what does the work* (`Model`/`Solver`), plus a great log/viewer.

**The robotics twist:** unlike LLM evals which have *one* swappable input (the
model), robotics evals have **two orthogonal swappable inputs**:

1. **The VLA / Policy** — the "brain". Maps observations + instruction → actions.
2. **The Embodiment** — the "body + world". A real robot or a simulator. Produces
   observations and consumes actions; owns the action/observation *spaces* and
   the success/reset machinery.

A benchmark `Task` is defined *independently* of both. The framework's job is to
(a) verify a given (policy, embodiment) pair is *compatible*, (b) run the
closed-loop rollout, (c) score it, and (d) log everything. This 2-input
factoring is the central architectural commitment and everything else follows
from it.

## 1. Scope of this first deliverable

Foundational framework only. Concretely:

- Core abstractions & interfaces (`Policy`, `Embodiment`, `Task`, `Scorer`,
  spaces, observation/action types).
- The rollout engine (`rollout()` — the closed control loop) and the top-level
  `eval()` entry point producing an immutable `EvalLog`.
- A **registry + decorators** (`@task`, `@policy`, `@embodiment`, `@scorer`) so
  third parties extend by registration, mirroring Inspect.
- **Mock policy + mock embodiment** ("CubePick" toy world) so the entire stack is
  testable with zero hardware/sim dependencies in CI.
- **Rerun logging** as an optional, pluggable sink (degrades gracefully when
  rerun isn't installed).
- **Structured eval logs** (JSON, versioned schema) + a tiny CLI (`robolens`).
- Full pytest suite, CI (GitHub Actions), packaging (`pyproject.toml`/uv), docs
  scaffold, contributor guidelines. Open-source best practices throughout.

**Out of scope for v0:** specific sim backends (ManiSkill/RoboSuite/Isaac/MuJoCo
adapters), specific VLA model weights (OpenVLA/π0/Octo adapters), a web viewer,
distributed/parallel rollout execution. These get their own plans later. We
design the *interfaces* so these slot in without refactoring.

## 2. Key assumptions (made autonomously — flagged for later confirmation)

- **Language:** Python 3.10+ (matches Inspect AI, robotics ecosystem, `uv`).
- **Numerics:** NumPy is the lingua franca for observations/actions. Torch is
  *not* a core dependency (policies may use it internally; framework stays
  framework-agnostic). Images are `np.ndarray` (H,W,C uint8).
- **Sync-first:** rollouts are synchronous (`step()`), because real robots are
  inherently sequential and most sims are too. Async is a later extension; we
  keep the interface `async`-friendly by not baking blocking assumptions into
  return types. (Revisit — see open questions.)
- **Primary use case is real-world benchmarks**, so the `Embodiment` interface is
  designed around a real robot's reality (no `reset()`-to-arbitrary-state
  guarantee, human-in-the-loop reset hooks, wall-clock control rate), with sims
  as a *stricter* special case that can offer more (seeding, deterministic
  reset, privileged success detection).

## 3. Architecture

### 3.1 Module layout

```
robolens/
  __init__.py          # public API surface (eval, rollout, decorators, types)
  _version.py
  types.py             # Observation, Action, StepResult, dataclasses
  spaces.py            # Space, Box, Dict, Discrete — compatibility checking
  policy.py            # Policy ABC/Protocol + PolicyInfo metadata
  embodiment.py        # Embodiment ABC/Protocol + EmbodimentInfo metadata
  task.py              # Task, Trial/Episode spec, success criteria hooks
  scorer.py            # Scorer protocol + builtin scorers + Score/Metric
  rollout.py           # rollout() closed-loop engine, RolloutResult
  eval.py              # eval() orchestration, EvalConfig, EvalLog (immutable)
  compat.py            # check_compatibility(policy, embodiment) -> report
  registry.py          # global registry + @task/@policy/@embodiment/@scorer
  logging/
    __init__.py
    sink.py            # LogSink protocol (start/log_step/finish hooks)
    rerun_sink.py      # RerunSink (optional import; no-op stub if missing)
    json_log.py        # EvalLog read/write, schema versioning
  mock/
    __init__.py
    cubepick.py        # CubePickEmbodiment (2D toy world, deterministic)
    policies.py        # ScriptedPolicy, RandomPolicy, NoopPolicy
  cli.py               # `robolens` CLI: list/run/inspect
  py.typed
```

### 3.2 Core data types (`types.py`, `spaces.py`)

```python
@dataclass(frozen=True)
class Observation:
    images: dict[str, np.ndarray]      # camera_name -> (H,W,C) uint8
    state: dict[str, np.ndarray]       # proprioception: joints, eef pose, gripper
    instruction: str | None            # language goal for this step (usually const)
    extra: dict[str, Any]              # embodiment-specific extras
    timestamp: float                   # wall-clock or sim time (seconds)

@dataclass(frozen=True)
class Action:
    data: np.ndarray                   # raw action vector
    space_id: str                      # which action space this targets
    meta: dict[str, Any]               # e.g. predicted by which VLA head

@dataclass(frozen=True)
class StepResult:
    observation: Observation           # observation AFTER applying the action
    reward: float | None               # optional dense/sparse signal (sims)
    terminated: bool                   # task ended (success or hard failure)
    truncated: bool                    # time/limit cut-off
    info: dict[str, Any]
```

`Space` types (`Box`, `Dict`, `Discrete`) describe action & observation shapes
and enable `compat.check_compatibility()` to fail *fast and loud* before a
rollout if a VLA emits 7-DoF actions but the arm expects 6-DoF, or expects a
wrist camera the embodiment doesn't have.

### 3.3 The two inputs

```python
class Policy(Protocol):              # the VLA / "brain"
    info: PolicyInfo                 # name, action_space it emits, obs it needs
    def reset(self, task: Task) -> None: ...
    def act(self, observation: Observation) -> Action: ...
    # optional: act_batch / async_act later

class Embodiment(Protocol):          # the robot or sim / "body + world"
    info: EmbodimentInfo             # name, action_space, observation_space,
                                     # control_hz, is_simulated, capabilities
    def reset(self, task: Task, *, seed: int | None = None) -> Observation: ...
    def step(self, action: Action) -> StepResult: ...
    def close(self) -> None: ...
    # optional sim-only: render(), set_state(), get_privileged_state()
```

`PolicyInfo`/`EmbodimentInfo` carry the *spaces* and *capabilities* used for
compatibility checking and logging. `capabilities` is a set of opt-in flags
(`"seedable"`, `"resettable"`, `"privileged_success"`, `"renderable"`) so the
framework and scorers can ask "can this embodiment do X?" rather than assuming.

### 3.4 Task & success criteria (`task.py`)

```python
@dataclass
class Task:
    name: str
    instruction: str                  # language goal given to the VLA
    scorer: Scorer | list[Scorer]
    max_steps: int                    # truncation horizon
    control_hz: float | None          # desired loop rate (None = as fast as possible)
    num_trials: int = 1               # episodes per (policy, embodiment) eval
    setup: Callable | None            # optional per-trial setup hook (e.g. human reset prompt)
    metadata: dict[str, Any] = field(default_factory=dict)
```

A `Task` is *embodiment-agnostic*: it describes the goal and how to score it, not
how the robot is built. Success detection is delegated to `Scorer`s, which may
use either privileged embodiment state (sim) or learned/sensor-based detectors
(real world).

### 3.5 Scoring (`scorer.py`)

Mirrors Inspect's `Scorer`/`Score`/`Metric` split.

```python
@dataclass(frozen=True)
class Score:
    value: float | bool | str         # primary outcome (e.g. success=True)
    explanation: str | None
    metadata: dict[str, Any]

class Scorer(Protocol):
    def __call__(self, trial: TrialRecord) -> Score: ...

# builtins: success_at_end, reached_goal_state, min_distance_to_goal,
#           episode_length, composite([...]) ; reducers across trials:
#           mean, success_rate, stderr, pass_at_k
```

A `Scorer` consumes the *recorded trajectory* (`TrialRecord`: every Observation,
Action, StepResult, timing) — not live env access — so scoring is reproducible
from a saved log and decoupled from the rollout. Sim "privileged success" is just
data in `StepResult.info` that a scorer reads.

### 3.6 Rollout engine (`rollout.py`)

The closed control loop — the heart of the framework:

```
policy.reset(task); obs = embodiment.reset(task, seed=...)
for t in range(task.max_steps):
    action = policy.act(obs)                 # VLA inference
    step = embodiment.step(action)           # robot/sim executes
    sink.log_step(t, obs, action, step)      # rerun + json
    record.append(...)
    if step.terminated or step.truncated: break
    obs = step.observation
    pace_to(task.control_hz)                  # honor control rate
return RolloutResult(record, ...)
```

Cross-cutting concerns handled here: control-rate pacing, exception capture
(a policy/robot exception is *recorded as a failed trial*, not a crash —
critical for unattended overnight benchmark runs), and per-step logging.

### 3.7 eval() & EvalLog (`eval.py`)

```python
def eval(
    tasks: Task | list[Task],
    policy: Policy,
    embodiment: Embodiment,
    *,
    log_dir: str = "logs",
    sinks: list[LogSink] | None = None,
    seed: int | None = None,
) -> EvalLog: ...
```

Steps: resolve tasks → `check_compatibility(policy, embodiment)` (raise on hard
mismatch, warn on soft) → for each task, for each trial, run `rollout()` → score
→ aggregate metrics → write immutable `EvalLog` (JSON, schema-versioned) to
`log_dir`. `EvalLog` mirrors Inspect: header (config, policy/embodiment info,
git rev, timestamps, package versions) + per-trial results + aggregate metrics.

### 3.8 Logging sinks (`logging/`)

`LogSink` protocol: `on_eval_start`, `on_trial_start`, `log_step`,
`on_trial_end`, `on_eval_end`. Two builtins:

- `JsonLogSink` — always on; the canonical reproducible record.
- `RerunSink` — optional. Logs camera images, 3D eef poses, joint time-series,
  action vectors, and success markers to a Rerun recording (`.rrd`). Imported
  lazily; if `rerun-sdk` isn't installed, RoboLens warns once and no-ops so core
  never hard-depends on it.

### 3.9 Registry & decorators (`registry.py`)

`@task`, `@policy`, `@embodiment`, `@scorer` register factories by name so the
CLI and third-party packages discover them (entry-points later). Exactly the
Inspect extension story.

### 3.10 CLI (`cli.py`)

`robolens list [tasks|policies|embodiments]`, `robolens run --task X --policy Y
--embodiment Z`, `robolens inspect <log.json>`. Thin wrapper over `eval()`.

## 4. Testing strategy (pytest, TDD)

- **Unit:** spaces & compatibility (matching, mismatching, subset cameras);
  types immutability; scorers on synthetic `TrialRecord`s; registry; json-log
  round-trip + schema version.
- **Integration:** full `eval()` on `CubePick` mock world with `ScriptedPolicy`
  (deterministic success) and `RandomPolicy` (mostly failure) → assert success
  rates, log structure, reproducibility under fixed seed.
- **Logging:** `RerunSink` is exercised behind a guard (skips if rerun missing);
  a fake sink verifies the hook sequence/ordering.
- **Property:** seeded determinism — same seed ⇒ identical `EvalLog` (modulo
  timestamps).
- Coverage gate in CI; `ruff` + `mypy` + `pytest` all green before merge.

## 5. Open questions (resolve via critique loop / later confirmation)

1. **Sync vs async rollout.** Sync now; is the interface future-proof for async
   real-robot drivers and batched sim envs? Confirm `Action`/`StepResult` don't
   preclude it.
2. **Action/observation space taxonomy.** Is `Box/Dict/Discrete` enough, or do we
   need explicit robot semantics (joint vs eef-delta vs eef-abs, gripper
   continuous vs binary)? Leaning toward a thin `semantics` tag on `Box` rather
   than a rigid enum.
3. **Vectorized/parallel embodiments** (N sim envs at once) — interface hook now
   or later? Leaning: design `rollout` to *not* assume single-env so a
   `VectorEmbodiment` slots in later.
4. **Real-world reset & safety.** How much of human-in-the-loop reset, e-stop,
   and safety-limit checking belongs in core vs adapters? v0: hooks only.
5. **Naming:** `Policy` vs `VLA` vs `Agent`; `Embodiment` vs `Env` vs `Robot`.
   Leaning `Policy`/`Embodiment` (precise, not over-loaded with RL `Env`).

## 6. Milestones (each its own commit/push; plan-per-feature as repo grows)

- **M0** Packaging, CI, license/readme/contributing, repo hygiene. *(this PR)*
- **M1** `types`, `spaces`, `compat` + tests.
- **M2** `Policy`/`Embodiment`/`Task`/`Scorer` interfaces + mock CubePick + tests.
- **M3** `rollout` + `eval` + `JsonLogSink` + `EvalLog` + integration tests.
- **M4** `registry` + decorators + CLI.
- **M5** `RerunSink` + visualization docs.
- **M6** Docs site scaffold + "write your first benchmark" tutorial.

Later plans (separate files): sim adapter (MuJoCo/ManiSkill), one real VLA
adapter, parallel rollout, web viewer.
```
