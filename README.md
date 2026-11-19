# jero-manager

<p align="center">
  <a href="https://hook.lma1980.duckdns.org/jobs">
    <img src="https://hook.lma1980.duckdns.org/cards/LMA1980/jero-manager/card.svg" alt="jero-manager Factory Scorecard" width="520"/>
  </a>
</p>

<p align="center">
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-manager/status.svg" alt="CI Status"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-manager/score.svg" alt="Quality Score"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-manager/coverage.svg" alt="Coverage"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-manager/benchmark.svg" alt="Benchmark"/></a>
</p>

______________________________________________________________________

Control plane for the SpacemiT K3 AI platform.

## Overview

`jero-manager` provides the orchestration layer, session management, content-addressed model store, and external APIs for the SpacemiT K3 platform:

- **Binaries**: `jero-manager` (control-plane daemon) and `jero-manager-gc` (content-addressed model catalog & garbage-collection CLI; Phase 1 alias `k3-store`).
- **IPC Daemon**: Apache Fory and Apache Arrow IPC serialization over Unix Domain Sockets (UDS) + `memfd` shared memory.
- **Persistence**: Active in-memory sessions and CAS blob/manifest catalogs backed by DuckDB (`DuckDbPersistenceStore`) for reboot persistence.
- **Modular WebAPI & Console Endpoints**: OpenAI- and Anthropic-compatible protocol endpoints alongside [`mu-auth`](https://github.com/LMA1980/mu-auth)-gated `/api/mr/v1/status` and `/api/mr/v1/connection` endpoints consumed by [`mu-console`](https://github.com/LMA1980/mu-console).
- **Observability & Continuous Delivery**: Streams W3C traces and `MUOB`-framed Apache Arrow IPC telemetry via [`mu-observable-ipc`](https://github.com/LMA1980/mu-observable-ipc), with deployment managed via [`mu-cd`](https://github.com/LMA1980/mu-cd) (`.mucd/mu-cd.yml`).
- **Model Subsystem**: Content-addressed store, model pull/search, and vendor adapters.

## Ecosystem & Documentation

- **Public Documentation Portal**: [`lma1980.github.io/model-runner`](https://github.com/LMA1980/lma1980.github.io/tree/main/model-runner) ([Reference Manual](https://github.com/LMA1980/lma1980.github.io/blob/main/model-runner/manual.md), [Crate Docs](https://github.com/LMA1980/lma1980.github.io/tree/main/model-runner/docs/crates))
- **Jero Dependencies & Companion Crates**: [`jero-core`](https://github.com/LMA1980/jero-core), [`jero-tokenizer`](https://github.com/LMA1980/jero-tokenizer), [`jero-kernels-rvv`](https://github.com/LMA1980/jero-kernels-rvv), [`jero-arch-mistral`](https://github.com/LMA1980/jero-arch-mistral), [`jero-kv-cache`](https://github.com/LMA1980/jero-kv-cache), and [`jero-engine`](https://github.com/LMA1980/jero-engine)
- **Platform Integrations**: [`mu-auth`](https://github.com/LMA1980/mu-auth), [`mu-console`](https://github.com/LMA1980/mu-console), [`mu-observable-ipc`](https://github.com/LMA1980/mu-observable-ipc), and [`mu-cd`](https://github.com/LMA1980/mu-cd)

## License

Licensed under the GNU Affero General Public License v3.0 (`AGPL-3.0-only`). See [LICENSE](LICENSE) for details.
