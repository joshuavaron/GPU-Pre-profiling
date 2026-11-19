# fakecuda

## Usage

```bash
make
./run <command>
make test
```

Environment variables:

- `FAKECUDA_DEVICE_COUNT`: number of virtual GPUs (default 3)
- `FAKECUDA_LOG=1`: log emulated calls to stderr
- `FAKECUDA_DARKAPI_TRAP=1`: trap stubs for unknown `cuGetExportTable` tables

## Files

- `fakecuda.h`: shared declarations
- `device.c`: virtual device table
- `runtime.c`: CUDA runtime API (`cuda*`)
- `driver.c`: CUDA driver API (`cu*`)
- `procaddr.c`: `cuGetProcAddress` table
- `darkapi.c`: `cuGetExportTable` tables
- `nvml.c`: NVML (`nvml*`)
- `memory.c`: device memory allocation
- `launch.c`: kernel registration and launch
- `stream.c`: streams and events
- `stubs.c`, `runtime_stubs.c`: fallbacks for functions that aren't implemented
- `libcudart.map`: symbol version script
- `run`: sets up the environment and runs a command under the shim
- `demo.py`: prints the virtual GPUs as PyTorch sees them
- `docs/`: notes
- `tests/`: tests
- `tools/gen_runtime_stubs.sh`: generates `runtime_stubs.c`
