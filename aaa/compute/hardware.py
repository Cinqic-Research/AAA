"""A reproducible, privacy-preserving probe of the machine AAA runs on.

The machine itself is the primary source. Every field records *how* it was
established:

``machine-verified``
    read from the running system by the named command or file;
``owner-declared``
    stated by the repository owner and not verifiable from software (for
    example a power supply or DIMM timings when DMI tables need root);
``unavailable``
    neither measured nor declared, with the reason.

Nothing identifying is recorded: no serial numbers, no GPU UUID, no MAC or IP
address, no hostname, and no username. Paths under the user's home directory or
removable-media mount are reduced to a placeholder, and a filesystem label is
kept only where it names the device's role (``Cinqic Storage``).
"""

from __future__ import annotations

import getpass
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROFILE_SCHEMA = "aaa.compute.hardware_profile.v1"

MACHINE = "machine-verified"
DECLARED = "owner-declared"
UNAVAILABLE = "unavailable"

OWNER_DECLARED: dict[str, dict[str, str]] = {
    "memory.module_configuration": {
        "value": "16 GB DDR4-3000 CL16",
        "note": "DIMM speed and timings need root-only DMI tables; recorded from docs/hardware.md",
    },
    "gpu.board_model": {
        "value": "Gigabyte GeForce RTX 2060 OC",
        "note": "the PCI subsystem vendor (Gigabyte) is machine-verified; the retail model name is not",
    },
    "platform.power_supply": {
        "value": "650 W",
        "note": "a power supply is not discoverable from software; recorded from docs/hardware.md",
    },
}


def _run(command: Sequence[str], timeout: float = 15.0) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    try:
        completed = subprocess.run(
            list(command), capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout if completed.returncode == 0 else None


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def field(value: Any, source: str, provenance: str = MACHINE, note: str | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"value": value, "provenance": provenance, "source": source}
    if note:
        entry["note"] = note
    return entry


def missing(source: str, reason: str) -> dict[str, Any]:
    return {"value": None, "provenance": UNAVAILABLE, "source": source, "note": reason}


def redact(text: str) -> str:
    """Remove the home directory and username from a path-like string."""

    home = str(Path.home())
    try:
        user = getpass.getuser()
    except Exception:
        user = ""
    out = text.replace(home, "<home>")
    if user:
        out = re.sub(rf"(?<=/){re.escape(user)}(?=/|$)", "<user>", out)
    return out


# ----------------------------------------------------------------------
# CPU
# ----------------------------------------------------------------------
def probe_cpu() -> dict[str, Any]:
    lscpu = _run(["lscpu"])
    info: dict[str, str] = {}
    if lscpu:
        for line in lscpu.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                info[key.strip()] = value.strip()
    flags = set(info.get("Flags", "").split())
    simd = [
        name
        for name in ("sse4_2", "avx", "avx2", "fma", "f16c", "bmi2", "avx512f", "sha_ni")
        if name in flags
    ]
    caches = {}
    cache_text = _run(["lscpu", "--caches"])
    if cache_text:
        for line in cache_text.splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 3:
                caches[parts[0]] = {"per_instance": parts[1], "total": parts[2]}

    def get(key: str) -> dict[str, Any]:
        return field(info[key], "lscpu") if key in info else missing("lscpu", f"{key} not reported")

    threads = os.cpu_count()
    cores_per_socket = info.get("Core(s) per socket")
    sockets = info.get("Socket(s)")
    physical = (
        int(cores_per_socket) * int(sockets) if cores_per_socket and sockets and sockets.isdigit() else None
    )
    return {
        "model": get("Model name"),
        "vendor": get("Vendor ID"),
        "architecture": get("Architecture"),
        "family_model_stepping": field(
            f"{info.get('CPU family')}/{info.get('Model')}/{info.get('Stepping')}",
            "lscpu",
            note="AMD family 25 model 80 is Zen 3 (Cezanne)",
        )
        if "CPU family" in info
        else missing("lscpu", "family not reported"),
        "physical_cores": field(physical, "lscpu (cores per socket x sockets)")
        if physical
        else missing("lscpu", "core count not reported"),
        "logical_threads": field(threads, "os.cpu_count / nproc"),
        "threads_per_core": get("Thread(s) per core"),
        "simd_flags": field(simd, "lscpu Flags (subset relevant to numerical kernels)"),
        "avx512": field("avx512f" in flags, "lscpu Flags"),
        "max_mhz": get("CPU max MHz"),
        "min_mhz": get("CPU min MHz"),
        "frequency_boost": get("Frequency boost"),
        "caches": field(caches, "lscpu --caches") if caches else missing("lscpu --caches", "not reported"),
        "numa_nodes": get("NUMA node(s)"),
    }


# ----------------------------------------------------------------------
# memory
# ----------------------------------------------------------------------
def probe_memory() -> dict[str, Any]:
    meminfo = _read("/proc/meminfo") or ""
    values: dict[str, int] = {}
    for line in meminfo.splitlines():
        match = re.match(r"^(\w+):\s+(\d+) kB", line)
        if match:
            values[match.group(1)] = int(match.group(2)) * 1024
    swaps = _read("/proc/swaps") or ""
    swap_entries = [
        {"type": parts[1], "size_bytes": int(parts[2]) * 1024}
        for parts in (line.split() for line in swaps.splitlines()[1:])
        if len(parts) >= 3
    ]
    dmi = _run(["dmidecode", "-t", "memory"])
    declared = OWNER_DECLARED["memory.module_configuration"]
    return {
        "total_usable_bytes": field(values.get("MemTotal"), "/proc/meminfo MemTotal"),
        "available_bytes_at_capture": field(values.get("MemAvailable"), "/proc/meminfo MemAvailable"),
        "swap": field(swap_entries, "/proc/swaps", note="swap file location omitted"),
        "swap_total_bytes": field(values.get("SwapTotal"), "/proc/meminfo SwapTotal"),
        "module_configuration": field(declared["value"], "docs/hardware.md", DECLARED, declared["note"])
        if dmi is None
        else field(dmi, "dmidecode -t memory"),
    }


# ----------------------------------------------------------------------
# GPU and CUDA
# ----------------------------------------------------------------------
_SMI_FIELDS = (
    "name",
    "memory.total",
    "driver_version",
    "pci.bus_id",
    "compute_cap",
    "power.limit",
    "power.max_limit",
    "pcie.link.gen.current",
    "pcie.link.gen.max",
    "pcie.link.width.current",
    "pcie.link.width.max",
    "clocks.max.sm",
    "clocks.max.memory",
    "vbios_version",
    "display_active",
)


def probe_gpu() -> dict[str, Any]:
    smi = _run(["nvidia-smi", f"--query-gpu={','.join(_SMI_FIELDS)}", "--format=csv,noheader,nounits"])
    header = _run(["nvidia-smi"])
    cuda_from_header = None
    if header:
        match = re.search(r"CUDA Version:\s*([\d.]+)", header)
        cuda_from_header = match.group(1) if match else None
    lspci = _run(["lspci", "-nnk"]) or ""
    display_adapters = [
        line.split(": ", 1)[1]
        for line in lspci.splitlines()
        if re.search(r"VGA compatible controller|3D controller", line)
    ]
    nvidia_subsystem = None
    lines = lspci.splitlines()
    for index, line in enumerate(lines):
        if "NVIDIA" in line and "VGA" in line:
            for follow in lines[index + 1 : index + 4]:
                if "Subsystem:" in follow:
                    nvidia_subsystem = follow.split("Subsystem:", 1)[1].strip()
    if not smi:
        return {
            "nvidia": missing("nvidia-smi", "nvidia-smi absent or failed"),
            "display_adapters": display_adapters,
        }
    values = [value.strip() for value in smi.strip().splitlines()[0].split(",")]
    row = dict(zip(_SMI_FIELDS, values, strict=False))
    compute_processes = _run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"])
    graphics_note = (
        "the RTX 2060 also drives the desktop display (Xorg/Cinnamon are resident on it), so a "
        "few hundred MiB of VRAM and some SM time are not available to AAA during measurements"
        if row.get("display_active", "").lower() == "enabled"
        else None
    )
    board = OWNER_DECLARED["gpu.board_model"]
    return {
        "name": field(row.get("name"), "nvidia-smi"),
        "pci_device": field(
            next((a for a in display_adapters if "NVIDIA" in a), None),
            "lspci -nnk",
            note="PCI IDs, no serial",
        ),
        "pci_subsystem": field(nvidia_subsystem, "lspci -nnk"),
        "board_model": field(board["value"], "docs/hardware.md", DECLARED, board["note"]),
        "memory_total_mib": field(int(row["memory.total"]), "nvidia-smi memory.total"),
        "nvidia_driver_version": field(row.get("driver_version"), "nvidia-smi driver_version"),
        "driver_supported_cuda": field(
            cuda_from_header,
            "nvidia-smi header",
            note="the newest CUDA runtime this driver supports; not an installed toolkit",
        ),
        "compute_capability": field(row.get("compute_cap"), "nvidia-smi compute_cap"),
        "power_limit_watts": field(row.get("power.limit"), "nvidia-smi power.limit"),
        "power_max_limit_watts": field(row.get("power.max_limit"), "nvidia-smi power.max_limit"),
        "pcie_link": field(
            {
                "generation_current": row.get("pcie.link.gen.current"),
                "generation_max": row.get("pcie.link.gen.max"),
                "width_current": row.get("pcie.link.width.current"),
                "width_max": row.get("pcie.link.width.max"),
            },
            "nvidia-smi pcie.link.*",
            note="read while idle; the link may train down when idle",
        ),
        "max_clocks_mhz": field(
            {"sm": row.get("clocks.max.sm"), "memory": row.get("clocks.max.memory")},
            "nvidia-smi clocks.max.*",
        ),
        "vbios_version": field(row.get("vbios_version"), "nvidia-smi vbios_version"),
        "display_attached": field(row.get("display_active"), "nvidia-smi display_active", note=graphics_note),
        "resident_compute_processes": field(
            len([line for line in (compute_processes or "").splitlines() if line.strip()]),
            "nvidia-smi --query-compute-apps",
        ),
        "display_adapters": field(display_adapters, "lspci -nnk", note="includes the unused integrated GPU"),
    }


def probe_cuda_stack() -> dict[str, Any]:
    """Driver, toolkit and Python CUDA runtime are three different things; keep them apart."""

    nvcc = _run(["nvcc", "--version"])
    toolkit_dirs = (
        sorted(str(path) for path in Path("/usr/local").glob("cuda*")) if Path("/usr/local").exists() else []
    )
    record: dict[str, Any] = {
        "nvcc": field(nvcc.strip().splitlines()[-1], "nvcc --version")
        if nvcc
        else field(None, "which nvcc", note="no CUDA toolkit compiler is installed"),
        "system_toolkit_directories": field(toolkit_dirs, "/usr/local/cuda*"),
        "driver_library": field(
            sorted(path.name for path in Path("/usr/lib/x86_64-linux-gnu").glob("libcuda.so.*")),
            "/usr/lib/x86_64-linux-gnu/libcuda.so.*",
        ),
    }
    try:
        import cupy

        runtime = cupy.cuda.runtime.runtimeGetVersion()
        record["python_backend"] = field(
            {
                "library": "cupy",
                "version": cupy.__version__,
                "cuda_runtime_in_use": f"{runtime // 1000}.{(runtime % 1000) // 10}",
                "nvrtc": ".".join(str(part) for part in cupy.cuda.nvrtc.getVersion()),
            },
            "cupy.cuda.runtime.runtimeGetVersion / nvrtc.getVersion",
            note="CUDA runtime components come from NVIDIA's PyPI wheels pinned in requirements-cuda-lock.txt",
        )
    except Exception as error:
        record["python_backend"] = missing("import cupy", f"{type(error).__name__}: {error}")
    return record


# ----------------------------------------------------------------------
# storage and platform
# ----------------------------------------------------------------------
def probe_storage() -> dict[str, Any]:
    lsblk = _run(["lsblk", "-J", "-b", "-o", "NAME,SIZE,TYPE,MODEL,ROTA,TRAN,FSTYPE,MOUNTPOINTS,LABEL"])
    devices = []
    if lsblk:
        for disk in json.loads(lsblk).get("blockdevices", []):
            if disk.get("type") != "disk":
                continue
            partitions = []
            for part in disk.get("children", []) or []:
                mounts = [redact(m) for m in (part.get("mountpoints") or []) if m]
                usage = None
                for mount in part.get("mountpoints") or []:
                    if mount:
                        try:
                            total, used, free = shutil.disk_usage(mount)
                            usage = {"total_bytes": total, "used_bytes": used, "free_bytes": free}
                        except OSError:
                            usage = None
                partitions.append(
                    {
                        "name": part.get("name"),
                        "size_bytes": part.get("size"),
                        "filesystem": part.get("fstype"),
                        "label": part.get("label"),
                        "mountpoints": mounts,
                        "usage_at_capture": usage,
                    }
                )
            devices.append(
                {
                    "name": disk.get("name"),
                    "model": disk.get("model"),
                    "size_bytes": disk.get("size"),
                    "rotational": bool(disk.get("rota")),
                    "transport": disk.get("tran"),
                    "partitions": partitions,
                }
            )
    home_fs = _run(["findmnt", "-n", "-o", "FSTYPE", "-T", str(Path.home())])
    return {
        "block_devices": field(devices, "lsblk -J (serials not requested)")
        if devices
        else missing("lsblk", "lsblk unavailable"),
        "home_filesystem": field(
            (home_fs or "").strip() or None,
            "findmnt -T <home>",
            note="an encrypted (ecryptfs) home adds per-I/O overhead to the repository checkout",
        ),
        "roles": field(
            {
                "system_and_development": "NVMe SSD root filesystem (OS, repository checkout, virtual environments)",
                "cinqic_working_storage": "HDD partition labelled 'Cinqic Storage' (larger AAA workloads, scratch)",
            },
            "lsblk model/label + docs/hardware.md role assignment",
            note="device models and labels are machine-verified; the role assignment is the owner's",
        ),
    }


def probe_platform() -> dict[str, Any]:
    os_release = _read("/etc/os-release") or ""
    pretty = next(
        (
            line.split("=", 1)[1].strip('"')
            for line in os_release.splitlines()
            if line.startswith("PRETTY_NAME=")
        ),
        None,
    )
    compiler = _run(["gcc", "--version"])
    glibc = _run(["ldd", "--version"])
    try:
        blas = np.__config__.CONFIG["Build Dependencies"]["blas"]
        blas_record: Any = {
            "name": blas.get("name"),
            "version": blas.get("version"),
            "configuration": blas.get("openblas configuration"),
        }
    except Exception as error:
        blas_record = {"unavailable": str(error)}
    board = {
        key: _read(f"/sys/class/dmi/id/{key}")
        for key in ("board_vendor", "board_name", "board_version", "bios_vendor", "bios_version", "bios_date")
    }
    psu = OWNER_DECLARED["platform.power_supply"]
    return {
        "operating_system": field(pretty, "/etc/os-release PRETTY_NAME"),
        "kernel": field(platform.release(), "uname -r"),
        "motherboard": field(
            {k: v or None for k, v in board.items() if k.startswith("board")},
            "/sys/class/dmi/id/board_* (serial not read)",
        ),
        "firmware": field(
            {k: v or None for k, v in board.items() if k.startswith("bios")}, "/sys/class/dmi/id/bios_*"
        ),
        "python": field(sys.version.split()[0], "sys.version"),
        "system_compiler": field(compiler.splitlines()[0] if compiler else None, "gcc --version"),
        "glibc": field(glibc.splitlines()[0] if glibc else None, "ldd --version"),
        "numpy": field(np.__version__, "numpy.__version__"),
        "numpy_blas": field(blas_record, "numpy.__config__"),
        "power_supply": field(psu["value"], "docs/hardware.md", DECLARED, psu["note"]),
    }


def probe(commit: str | None = None) -> dict[str, Any]:
    """The full machine profile."""

    return {
        "schema": PROFILE_SCHEMA,
        "machine": "FLOWBOX",
        "captured_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_commit": commit,
        "privacy": (
            "no serial numbers, GPU UUID, MAC/IP address, hostname or username; home and removable-media "
            "paths are reduced to <home>/<user>"
        ),
        "provenance_legend": {
            MACHINE: "read from the running system by the named source",
            DECLARED: "stated by the owner; not verifiable from software",
            UNAVAILABLE: "neither measured nor declared",
        },
        "cpu": probe_cpu(),
        "memory": probe_memory(),
        "gpu": probe_gpu(),
        "cuda_stack": probe_cuda_stack(),
        "storage": probe_storage(),
        "platform": probe_platform(),
        "probes": [
            "lscpu",
            "lscpu --caches",
            "os.cpu_count",
            "/proc/meminfo",
            "/proc/swaps",
            "nvidia-smi (query and header)",
            "lspci -nnk",
            "nvcc --version",
            "cupy runtime queries",
            "lsblk -J",
            "findmnt",
            "shutil.disk_usage",
            "/etc/os-release",
            "/sys/class/dmi/id/{board,bios}_*",
            "gcc --version",
            "ldd --version",
            "numpy.__config__",
        ],
    }


def provenance_counts(profile: Mapping[str, Any]) -> dict[str, int]:
    """How many leaf fields carry each provenance label (a summary for the docs)."""

    counts: dict[str, int] = {MACHINE: 0, DECLARED: 0, UNAVAILABLE: 0}

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            if "provenance" in node and node["provenance"] in counts:
                counts[node["provenance"]] += 1
                return
            for value in node.values():
                walk(value)

    walk(profile)
    return counts
