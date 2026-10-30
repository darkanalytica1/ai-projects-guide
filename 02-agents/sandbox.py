"""Run untrusted, model-written Python somewhere that is NOT your laptop."""
import os
import pathlib
import subprocess
import tempfile


def run_in_e2b(code: str, timeout: int = 60) -> str:
    from e2b_code_interpreter import Sandbox  # needs E2B_API_KEY in the environment
    # isolated microVM; lifetime capped at 5 min, no internet from inside
    sbx = Sandbox.create(timeout=300, allow_internet_access=False)
    try:
        ex = sbx.run_code(code, timeout=timeout)
        out = "".join(ex.logs.stdout) + "".join(ex.logs.stderr)
        if ex.error:
            out += f"\nERROR {ex.error.name}: {ex.error.value}"
        return out[-4000:] or "(no output)"
    finally:
        sbx.kill()                            # always destroy the VM


def run_in_docker(code: str, timeout: int = 60) -> str:
    """Throwaway container: no network, 512 MB RAM, 1 CPU, read-only code mount."""
    with tempfile.TemporaryDirectory() as d:
        pathlib.Path(d, "main.py").write_text(code)
        cmd = ["docker", "run", "--rm", "--network", "none", "--memory", "512m",
               "--cpus", "1", "--pids-limit", "128", "-v", f"{d}:/work:ro",
               "python:3.11-slim", "python", "/work/main.py"]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return "ERROR: timed out"
        return ((p.stdout + p.stderr)[-4000:]) or "(no output)"


def run_code(code: str) -> str:
    backend = os.getenv("SANDBOX", "e2b").lower()
    return run_in_docker(code) if backend == "docker" else run_in_e2b(code)
