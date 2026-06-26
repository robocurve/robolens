# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the version is
`0.x`, breaking changes may occur on any minor release.

## [Unreleased]

### Added

- **Core framework foundation.** The two-input model for robotics evals:
  `Policy` (VLA) and `Embodiment` (real robot or simulator), with a benchmark
  `Task` defined independently of both.
- **Types & spaces:** `Observation`, `Action`, `ActionChunk` (open-loop chunked
  execution), `StepResult`; `Box`/`ObservationSpace`, `ActionSemantics`, and a
  canonical proprioception `StateSpec` vocabulary.
- **Scenes & scoring:** `Scene`/`Target` datasets; `Scorer`/`Score` with an
  epoch-reducer split (`mean`/`median`/`max`/`min`/`mode`/`pass_at_k`); builtin
  scorers including `success_at_end`, `min_distance_to_goal`, `reached_goal_state`,
  and an operator-verdict scorer; reserved `VLMScorer` interface.
- **Rollout engine:** open-loop chunk execution via a composable `Controller`
  middleware layer (`DefaultController`, `SmoothingController`,
  `EnsemblingController` for ACT/ALOHA temporal ensembling); an `Approver`
  safety gate (`AutoApprover`, `ClampApprover`); an error taxonomy
  (`PolicyError` continue vs `EmbodimentFault`/`SafetyAbort` halt); a typed
  transcript; per-trial seeding; and a `FrameStore` that streams frames to disk.
- **Compatibility checking:** fail-fast action/observation/semantics checks with
  key remapping, control-rate reconciliation, and scene realizability.
- **`eval()` / `eval_set()`:** Inspect-style orchestration returning immutable,
  schema-versioned `EvalLog`s with `fail_on_error` semantics; atomic JSON logs
  with a read-back guarantee; optional frame side-cars.
- **Registry & plugins:** decorators and `importlib.metadata` entry-point
  discovery so out-of-tree backends register without being imported.
- **Logging sinks:** canonical `JsonLogSink`; optional, lazily-imported
  `RerunSink` for [Rerun](https://github.com/rerun-io/rerun) visualization.
- **CLI:** `robolens list`, `robolens run`, and `robolens inspect <log>`.
- **String resolution:** `eval()`/`eval_set()` accept registry names
  (`eval("cubepick-reach", "scripted", "cubepick")`) in addition to objects.
- Dependency-free `CubePick` mock world and scripted/random/noop policies.

[Unreleased]: https://github.com/robocurve/robolens/commits/main
