import asyncio
import glob
import json
import os
from contextlib import asynccontextmanager

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

FAN_MIN_PERCENT = int(os.environ.get("FAN_MIN_PERCENT", "20"))
SAFETY_TEMP_C = float(os.environ.get("SAFETY_TEMP_C", "90"))
CRITICAL_TEMP_C = float(os.environ.get("CRITICAL_TEMP_C", "100"))
GPU_NAME = os.environ.get("GPU_NAME", "AMD Radeon Instinct MI50 (32GB)")

# The MI50's own fan header is unused (the card is passive); the physical
# fan that cools it is wired into a motherboard fan header instead, driven
# by the Super I/O chip (nct6775 family), not by amdgpu. Which chip channel
# (fan1/pwm1 vs fan2/pwm2 ...) maps to which physical header is board- and
# wiring-specific and was confirmed by hand on this machine; if the cabling
# changes, re-confirm and update these two values.
BOARD_HWMON_NAME = os.environ.get("BOARD_HWMON_NAME", "nct6797")
GPU_FAN_CHANNEL = int(os.environ.get("GPU_FAN_CHANNEL", "2"))
CPU_FAN_CHANNEL = int(os.environ.get("CPU_FAN_CHANNEL", "1"))

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "2"))

gpu = None
board = None  # {"hwmon": path, "auto_enable_value": <int captured at startup>}
state_lock = asyncio.Lock()
last_error = None

# model name -> {"status": str, "completed": int, "total": int, "done": bool, "error": str|None}
pull_jobs: dict[str, dict] = {}


def find_gpu():
    for device in sorted(glob.glob("/sys/class/drm/card*/device")):
        vram_total = os.path.join(device, "mem_info_vram_total")
        if not os.path.exists(vram_total):
            continue
        hwmon_dirs = glob.glob(os.path.join(device, "hwmon", "hwmon*"))
        if not hwmon_dirs:
            continue
        return {"device": device, "hwmon": hwmon_dirs[0]}
    return None


def find_board():
    for hwmon_dir in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        name_path = os.path.join(hwmon_dir, "name")
        if os.path.exists(name_path) and read_str(name_path) == BOARD_HWMON_NAME:
            enable_path = os.path.join(hwmon_dir, f"pwm{GPU_FAN_CHANNEL}_enable")
            auto_value = read_int(enable_path) if os.path.exists(enable_path) else 5
            return {"hwmon": hwmon_dir, "auto_enable_value": auto_value}
    return None


def read_int(path):
    with open(path) as f:
        return int(f.read().strip())


def read_str(path):
    with open(path) as f:
        return f.read().strip()


def write_str(path, value):
    with open(path, "w") as f:
        f.write(str(value))


def get_temps(hwmon):
    temps = {}
    for label_path in glob.glob(os.path.join(hwmon, "temp*_label")):
        idx = os.path.basename(label_path)[len("temp"):-len("_label")]
        label = read_str(label_path)
        input_path = os.path.join(hwmon, f"temp{idx}_input")
        if os.path.exists(input_path):
            temps[label] = read_int(input_path) / 1000.0
    return temps


def get_stats():
    device = gpu["device"]
    gpu_hwmon = gpu["hwmon"]

    vram_total = read_int(os.path.join(device, "mem_info_vram_total"))
    vram_used = read_int(os.path.join(device, "mem_info_vram_used"))
    gpu_busy_path = os.path.join(device, "gpu_busy_percent")

    result = {
        "name": GPU_NAME,
        "temps": get_temps(gpu_hwmon),
        "vram_used": vram_used,
        "vram_total": vram_total,
        "gpu_busy_percent": read_int(gpu_busy_path) if os.path.exists(gpu_busy_path) else None,
        "fan_min_percent": FAN_MIN_PERCENT,
        "safety_temp_c": SAFETY_TEMP_C,
        "critical_temp_c": CRITICAL_TEMP_C,
        "board_fan_available": board is not None,
    }

    if board is not None:
        bh = board["hwmon"]
        gpu_pwm = read_int(os.path.join(bh, f"pwm{GPU_FAN_CHANNEL}"))
        gpu_pwm_enable = read_int(os.path.join(bh, f"pwm{GPU_FAN_CHANNEL}_enable"))
        gpu_fan_rpm_path = os.path.join(bh, f"fan{GPU_FAN_CHANNEL}_input")
        cpu_fan_rpm_path = os.path.join(bh, f"fan{CPU_FAN_CHANNEL}_input")
        result.update({
            "fan_rpm": read_int(gpu_fan_rpm_path) if os.path.exists(gpu_fan_rpm_path) else None,
            "fan_percent": round(gpu_pwm / 255 * 100),
            "fan_mode": "manual" if gpu_pwm_enable == 1 else "auto",
            "cpu_fan_rpm": read_int(cpu_fan_rpm_path) if os.path.exists(cpu_fan_rpm_path) else None,
        })

    return result


def set_pwm(percent):
    bh = board["hwmon"]
    pwm_val = round(max(0, min(100, percent)) / 100 * 255)
    write_str(os.path.join(bh, f"pwm{GPU_FAN_CHANNEL}_enable"), "1")
    write_str(os.path.join(bh, f"pwm{GPU_FAN_CHANNEL}"), pwm_val)


def set_auto():
    bh = board["hwmon"]
    write_str(os.path.join(bh, f"pwm{GPU_FAN_CHANNEL}_enable"), board["auto_enable_value"])


async def get_ollama_status():
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            version_r, ps_r, tags_r = await asyncio.gather(
                client.get(f"{OLLAMA_URL}/api/version"),
                client.get(f"{OLLAMA_URL}/api/ps"),
                client.get(f"{OLLAMA_URL}/api/tags"),
            )
        return {
            "available": True,
            "version": version_r.json().get("version"),
            "running": ps_r.json().get("models", []),
            "installed": tags_r.json().get("models", []),
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


MODEL_LOAD_TIMEOUT = float(os.environ.get("MODEL_LOAD_TIMEOUT", "300"))


async def set_model_keep_alive(model: str, keep_alive):
    """keep_alive=0 unloads the model now; a duration string (e.g. "10m") loads it.

    Large models can take well over a minute to read off disk into VRAM, so
    this needs a generous timeout rather than a short request-latency one.
    """
    async with httpx.AsyncClient(timeout=MODEL_LOAD_TIMEOUT) as client:
        r = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": keep_alive, "stream": False},
        )
        r.raise_for_status()
        return r.json()


async def run_pull_job(model: str):
    pull_jobs[model] = {"status": "starting", "completed": 0, "total": 0, "done": False, "error": None}
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST", f"{OLLAMA_URL}/api/pull", json={"model": model, "stream": True}
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if "error" in data:
                        pull_jobs[model] = {**pull_jobs[model], "error": data["error"], "done": True}
                        return
                    pull_jobs[model] = {
                        "status": data.get("status", ""),
                        "completed": data.get("completed", 0),
                        "total": data.get("total", 0),
                        "done": data.get("status") == "success",
                        "error": None,
                    }
    except Exception as exc:
        pull_jobs[model] = {**pull_jobs.get(model, {}), "error": str(exc), "done": True}


# This scrapes ollama.com's public search page HTML because Ollama does not
# publish a JSON API for browsing the model library (only for pulling a tag
# you already know). It's best-effort: if ollama.com changes its markup,
# this stops returning results until updated.
async def search_ollama_library(query: str):
    async with httpx.AsyncClient(timeout=8) as client:
        r = await client.get("https://ollama.com/search", params={"q": query})
        r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    results = []
    seen = set()
    for a in soup.select("a[href^='/library/']"):
        name = a["href"].removeprefix("/library/")
        if not name or "/" in name or name in seen:
            continue
        seen.add(name)
        h2 = a.find("h2")
        p = a.find("p")
        tags_div = a.find("div", class_="mt-3")
        tags = [s.get_text(strip=True) for s in tags_div.find_all("span")] if tags_div else []
        downloads_el = a.find(attrs={"title": lambda v: v and "download" in v})
        downloads = downloads_el.get_text(strip=True) if downloads_el else ""
        results.append({
            "name": (h2.get_text(strip=True) if h2 else name),
            "description": p.get_text(strip=True) if p else "",
            "tags": tags[:6],
            "downloads": downloads,
        })
        if len(results) >= 20:
            break
    return results


async def watchdog():
    """Overrides manual fan control if the GPU gets dangerously hot.

    Also re-probes for the GPU/board sensors on every tick when missing, so
    a slow-to-load hwmon driver (e.g. nct6775 racing with container start on
    boot) self-heals within one tick instead of needing a container restart.
    """
    global last_error, gpu, board
    while True:
        await asyncio.sleep(5)
        if gpu is None:
            gpu = find_gpu()
        if board is None:
            board = find_board()
        if gpu is None or board is None:
            continue
        try:
            async with state_lock:
                stats = get_stats()
                junction = stats["temps"].get("junction")
                if junction is None:
                    continue
                if stats["fan_mode"] == "manual" and junction >= CRITICAL_TEMP_C:
                    set_auto()
                elif stats["fan_mode"] == "manual" and junction >= SAFETY_TEMP_C and stats["fan_percent"] < 100:
                    set_pwm(100)
        except Exception as exc:
            last_error = str(exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global gpu, board
    gpu = find_gpu()
    board = find_board()
    task = asyncio.create_task(watchdog())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


class FanRequest(BaseModel):
    mode: str
    percent: int | None = None


@app.get("/api/stats")
async def api_stats():
    if gpu is None:
        raise HTTPException(503, "No AMD GPU with hwmon detected on this host")
    async with state_lock:
        try:
            stats = get_stats()
        except PermissionError as exc:
            raise HTTPException(500, f"Permission denied reading sysfs: {exc}")
    stats["ollama"] = await get_ollama_status()
    return stats


@app.post("/api/fan")
async def api_set_fan(req: FanRequest):
    if gpu is None:
        raise HTTPException(503, "No AMD GPU with hwmon detected on this host")
    if board is None:
        raise HTTPException(
            503,
            f"Motherboard sensor chip '{BOARD_HWMON_NAME}' not found. The GPU's "
            "fan is wired to a motherboard header, not the GPU's own header, "
            "so fan control needs that chip's driver loaded on the host "
            "(modprobe nct6775).",
        )
    async with state_lock:
        try:
            if req.mode == "auto":
                set_auto()
            elif req.mode == "manual":
                percent = req.percent if req.percent is not None else 50
                percent = max(FAN_MIN_PERCENT, min(100, percent))
                set_pwm(percent)
            else:
                raise HTTPException(400, "mode must be 'auto' or 'manual'")
            return get_stats()
        except PermissionError as exc:
            raise HTTPException(
                500,
                f"Permission denied writing sysfs ({exc}). The container needs "
                "write access to the host's /sys mount, apparmor:unconfined, "
                "and must run as root.",
            )


class ModelRequest(BaseModel):
    model: str


@app.post("/api/models/load")
async def api_model_load(req: ModelRequest):
    try:
        await set_model_keep_alive(req.model, "30m")
        return {"ok": True}
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, exc.response.text)
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Could not reach Ollama: {exc or exc.__class__.__name__}")


@app.post("/api/models/eject")
async def api_model_eject(req: ModelRequest):
    try:
        await set_model_keep_alive(req.model, 0)
        return {"ok": True}
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, exc.response.text)
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Could not reach Ollama: {exc or exc.__class__.__name__}")


@app.post("/api/models/pull")
async def api_model_pull(req: ModelRequest):
    existing = pull_jobs.get(req.model)
    if existing and not existing["done"]:
        raise HTTPException(409, "This model is already being pulled")
    asyncio.create_task(run_pull_job(req.model))
    return {"started": True}


@app.get("/api/models/pull/status")
async def api_model_pull_status():
    return pull_jobs


@app.get("/api/models/search")
async def api_model_search(q: str):
    if not q.strip():
        return {"results": []}
    try:
        return {"results": await search_ollama_library(q.strip())}
    except Exception as exc:
        raise HTTPException(502, f"ollama.com 模型庫搜尋失敗: {exc}")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
