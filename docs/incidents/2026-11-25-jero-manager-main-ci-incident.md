# Incident Report: `main` Branch CI Failure in `jero-manager` (`coverage-check` Worker Spawn Contention)

- **Date**: 2026-11-25
- **Repository**: `LMA1980/jero-manager`
- **Affected Commit**: `d5ff10903a22b9abc287d6443da9b65f07378786`
- **Tracking Issue**: [#122](https://github.com/LMA1980/jero-manager/issues/122)
- **Resolution Branch**: `feat/fix-main-ci-supervisor-llvm-cov-binary`

______________________________________________________________________

## 1. Summary & Impact

The `main` branch CI pipeline (`job-1795574580-d5ff1090`) on `LMA1980/jero-manager` failed `coverage-check` (`cargo llvm-cov`, `exit_code = 101`) in `tests/adversarial_edge_cases_test.rs`:

```text
spawn_worker took too long on premature exit: 2.398815177s
thread 'test_adversarial_worker_premature_exit_fast_detection' panicked at tests/adversarial_edge_cases_test.rs:103:5:
```

______________________________________________________________________

## 2. Root Cause Analysis

1. **`llvm-cov-target` Worker Binary Shadowing in Shared Target Directory**:
   `find_worker_binary_with()` in `src/supervisor.rs` walks ancestors of `std::env::current_exe()`. Under `cargo llvm-cov` on CI workers using a shared `CARGO_TARGET_DIR`, `current_exe` resides inside `.../llvm-cov-target/debug/build/jero-manager/...`. When an upstream `jero-engine` coverage job had recently populated `.../llvm-cov-target/debug/jero-engine`, ancestor traversal resolved that coverage-instrumented `jero-engine` binary ahead of the uninstrumented `.../debug/jero-engine` binary, causing spawned worker processes to inherit `LLVM_PROFILE_FILE`, emit foreign `.profraw` counters, and incur instrumentation and profile-flush overhead.
1. **Parallel Worker Spawn Contention**:
   `tests/adversarial_edge_cases_test.rs` launches multiple tests concurrently (`test_adversarial_worker_premature_exit_fast_detection`, `test_adversarial_context_limit_rejection`, and `test_adversarial_signal_reporting_and_recover_running_worker`) that each spawn `jero-engine` and multi-threaded Tokio runtimes simultaneously. Under `cargo llvm-cov` instrumentation, `jero-engine` startup and error exit stalled to `2.394s` before printing `[jero-engine error]`, while `spawn_and_attach_process` detected the exit within ~5ms (`2.3988s` total), exceeding the `2s` test bound.
1. **How the Failure Reached `main`**:
   Prior to merging PR `#121` (`4c131d09`), `test-check` and `coverage-check` passed before `jero-engine` `main` CI populated `.../llvm-cov-target/debug/jero-engine` in the shared CI target directory immediately preceding `jero-manager` `main` job `job-1795574580-d5ff1090`.

______________________________________________________________________

## 3. Remediation & Preventive Measures

### Remediation

1. Updated `find_worker_binary_with()` in `src/supervisor.rs` to skip `llvm-cov-target` ancestor candidates during the primary discovery pass and fall back to them only if no uninstrumented worker binary exists, backed by unit tests.
1. Added `WORKER_SPAWN_LOCK` in `tests/adversarial_edge_cases_test.rs` to serialize external `jero-engine` spawns across those integration tests while keeping the strict `< 2s` premature-exit assertion intact.
1. Verified the full Pre-PR Local CI Gate Sequence (`cargo fmt`, `mdformat`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`, and `cargo +nightly miri test --workspace --lib`).

### Preventive Measures

1. Ensure binary discovery helpers in shared CI target directories prefer uninstrumented `debug`/`release` artifacts over `llvm-cov-target` build artifacts so coverage runs remain isolated from external binaries.
1. Serialize timing-sensitive subprocess-spawning tests within integration test suites that launch full worker runtimes concurrently under instrumentation.
