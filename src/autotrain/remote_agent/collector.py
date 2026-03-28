"""GPU metrics collection using pynvml — sub-millisecond, no subprocess."""

from __future__ import annotations


def collect_gpu_metrics() -> list[dict]:
    """Query all GPUs via NVML. Returns list of metric dicts."""
    try:
        import pynvml  # noqa: works with both nvidia-ml-py and pynvml
    except ImportError:
        return _fallback_nvidia_smi()

    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus = []

        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU,
                )
            except pynvml.NVMLError:
                temp = None

            gpus.append({
                "gpu_index": i,
                "utilization_pct": float(util.gpu),
                "memory_used_mb": round(mem.used / 1048576, 1),
                "memory_total_mb": round(mem.total / 1048576, 1),
                "temperature_c": float(temp) if temp is not None else None,
            })

        pynvml.nvmlShutdown()
        return gpus

    except pynvml.NVMLError:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        return _fallback_nvidia_smi()


def _fallback_nvidia_smi() -> list[dict]:
    """Fallback to nvidia-smi subprocess if pynvml unavailable."""
    import subprocess

    try:
        result = subprocess.run(
            "nvidia-smi "
            "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu "
            "--format=csv,noheader,nounits",
            shell=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []

        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            gpus.append({
                "gpu_index": int(parts[0]),
                "utilization_pct": float(parts[1]),
                "memory_used_mb": float(parts[2]),
                "memory_total_mb": float(parts[3]),
                "temperature_c": float(parts[4]),
            })
        return gpus
    except Exception:
        return []
