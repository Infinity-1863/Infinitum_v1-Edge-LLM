# Runtime

The modified llama.cpp runtime is included at `vendor/llama.cpp`.

## Android

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build-runtime-android.ps1
```

Output:

```text
artifacts/runtime-android-arm64/bin/
```

## PC

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build-runtime-pc.ps1
```

Output:

```text
artifacts/runtime-pc/bin/
```

## Run

Use the built `bin` directory with `scripts/run-phone.ps1`, `scripts/run-pc.ps1`,
`scripts/bench-phone.ps1`, or `scripts/bench-pc.ps1`.
