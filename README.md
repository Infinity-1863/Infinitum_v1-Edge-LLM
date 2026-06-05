# Infinitum Edge LLM

Small Windows-first toolkit for running large open-weight models on a PC or an
Android phone with a `core + external experts` package layout.

This repository does **not** include model weights, API keys, or device-specific
secrets. It gives maintainers a clean public starting point for:

- organizing already-built model artifacts into a portable package;
- building the bundled modified llama.cpp runtime;
- launching a llama.cpp-compatible server on a PC;
- pushing the package to Android with ADB and launching a phone server;
- sharing reproducible dry-run commands before touching real hardware.

## Package Layout

```text
package/
  package.manifest.json
  model/
    core-model.gguf
  moe/
    expert-store-index.json
  moe_ggml_pack/
    experts.ggml_f16.bin
```

`model/` holds the resident core GGUF. `moe/` holds the router/expert metadata.
`moe_ggml_pack/` holds the external expert pack that can be memory-mapped or
cached by a patched runtime.

## Quick Start

Create a package manifest and copy/link the artifacts:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\split-model.ps1 `
  -ModelDir C:\Models\my-large-model `
  -OutDir .\package `
  -CoreFile C:\Models\my-large-model\core-model.gguf `
  -ExpertIndex C:\Models\my-large-model\expert-store-index.json `
  -ExpertPack C:\Models\my-large-model\experts.ggml_f16.bin `
  -LinkMode Copy
```

Build the bundled runtime for Android:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build-runtime-android.ps1
```

Build the bundled runtime for PC. The modified llama.cpp source is included in
this repository under `vendor/llama.cpp`.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build-runtime-pc.ps1
```

Intel oneAPI/SYCL build:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build-runtime-pc.ps1 -Backend sycl
```

Preview a PC launch:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-pc.ps1 `
  -PackageDir .\package `
  -ServerBin .\artifacts\runtime-pc\bin\llama-server.exe `
  -Port 18082 `
  -DryRun
```

Preview an Android phone launch:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-phone.ps1 `
  -PackageDir .\package `
  -AndroidBinDir .\artifacts\runtime-android-arm64\bin `
  -DeviceDir /data/local/tmp/infinitum-edge-llm `
  -PcPort 18080 `
  -PhonePort 8080 `
  -DryRun
```

Remove `-DryRun` only after the printed command plan looks correct.

Send a GPT-OSS chat message through the running server:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\chat.ps1 `
  -ServerUrl http://127.0.0.1:18080 `
  -Message "привет"
```

Measure a PC baseline or universal page-prefetch profile:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bench-pc.ps1 `
  -PackageDir .\package `
  -ServerBin .\artifacts\runtime-pc\bin\llama-server.exe `
  -Profile page-prefetch
```

Measure an already-pushed Android package:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bench-phone.ps1 `
  -DeviceDir /data/local/tmp/infinitum-edge-llm `
  -Profile page-prefetch `
  -PcPort 18110
```

## Notes

- Durable notes and benchmark history live in [docs/INDEX.md](docs/INDEX.md).
- Runtime source lives in [vendor/llama.cpp](vendor/llama.cpp). Build notes live
  in [docs/RUNTIME.md](docs/RUNTIME.md).
- `split-model.ps1` does not convert Hugging Face tensors into GGUF. Use your
  own converter or patched llama.cpp tooling first, then package the outputs.
- `build-expert-pack.ps1` can build a public portable expert pack from
  Hugging Face safetensors for inspection and experimentation.
- `run-pc.ps1` and `run-phone.ps1` use the bundled modified llama.cpp runtime,
  or any compatible `llama-server` that understands the external expert
  environment variables shown in the dry run.
- Android mode expects `adb` in PATH. Pass `-AndroidBinDir` to push a local
  Android `llama-server` bundle into the phone package `bin/` directory.
- For GPT-OSS chat, use `chat.ps1`. It sends the request as UTF-8 and uses the
  model's Harmony-style tokens instead of the generic ChatML template.
- `bench-pc.ps1` compares the plain launch path with a universal page-prefetch
  profile. It asks the runtime to warm one routed expert for the next layer
  through the OS page cache, without copying the whole expert pack into process
  RAM. Increase `-PrefetchMaxExperts` only when disk latency dominates.
- `bench-phone.ps1` does the same comparison on Android for a package that is
  already on the device. It starts the server detached, forwards the port,
  sends a UTF-8 GPT-OSS prompt, records timings/RSS, and stops the server.
- Keep raw model weights, logs, and device dumps out of public commits unless
  their licenses explicitly allow redistribution.

## Verify

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tests\smoke.ps1
```

## License

MIT. See [LICENSE](LICENSE).
