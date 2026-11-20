# Incident Report: `main` Branch CI Failure in `jero-manager` (`coverage-check` Race Flake)

- **Date**: 2026-11-19
- **Repository**: `LMA1980/jero-manager`
- **Affected Commit**: `fe1dcab9560419e411ad86afce8f2cb0fa1bdaaa`
- **Tracking Issue**: [#98](https://github.com/LMA1980/jero-manager/issues/98)
- **Resolution Branch**: `feat/fix-main-ci-lending-race-test`

______________________________________________________________________

## 1. Summary & Impact

The `main` branch CI pipeline (`job-1795073535-fe1dcab9`) on `LMA1980/jero-manager` failed `coverage-check` (`cargo llvm-cov`, `exit_code = 101`) at `tests/empirical_challenge_m1_lending_immutability.rs:452:5`:

```text
thread 'test_challenge2a_concurrent_race_condition_instant_reclaim' panicked at tests/empirical_challenge_m1_lending_immutability.rs:452:5:
borrower must have completed passes during race conditions
```

______________________________________________________________________

## 2. Root Cause Analysis

1. **Missing Thread Start Synchronization & Tight Owner Loop**:
   `test_challenge2a_concurrent_race_condition_instant_reclaim` spawned `borrower_handle` and `owner_handle` without a barrier, and `owner_handle` ran 100 `schedule_forward_pass` / `record_pass_completion` rounds in a tight loop with only `thread::yield_now()`.
1. **Starvation Under `cargo llvm-cov` Instrumentation**:
   Under `cargo llvm-cov` instrumentation overhead on CI workers, `owner_handle` could complete all 100 rounds before `borrower_handle` finished a single forward pass, immediately setting `stop_signal = true` and tripping `assert!(borrower_passes_completed > 0)` without exercising concurrent lending and reclaim.
1. **How the Regression Reached `main`**:
   Uninstrumented `cargo test` runs on lightly loaded hosts usually interleaved `borrower_handle` quickly enough to pass, masking the missing thread synchronization until `cargo llvm-cov` ran on `main`.

______________________________________________________________________

## 3. Remediation & Preventive Measures

### Remediation

1. Added a `std::sync::Barrier` (`Barrier::new(2)`) so `borrower_handle` and `owner_handle` start racing together, and added a `50us` idle window between `owner_handle` passes so `borrower_handle` deterministically executes concurrently, borrows idle cores, and exercises instant pass-boundary reclaim while preserving every assertion intact.
1. Verified the full Pre-PR Local CI Gate Sequence (`cargo fmt`, `cargo +1.100.0 clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`, and `cargo +nightly miri test --workspace --lib`).

### Preventive Measures

1. Synchronize concurrent test workers with explicit barriers and bounded inter-round windows rather than relying on OS thread startup timing or `thread::yield_now()` alone.
