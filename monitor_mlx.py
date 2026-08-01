#!/usr/bin/env python3
"""
MLX 训练监控：显存(Metal 统一内存) + 温度 + 降频信号
- 显存: 训练进程的 MLX Peak mem 从 pipeline_output.log 提取（mlx.core 内存 API 是进程内的，读不到别的进程）
- 温度: 尝试 sudo powermetrics 读取 GPU die temperature，sudo 不可用时自动降级为 It/sec 降频信号
- 告警: 显存>19GB / It/sec 较基线掉30% / 温度>85°C
用法: nohup venv/bin/python3 monitor_mlx.py >/dev/null 2>&1 &
"""
import os, re, subprocess, time
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent
PIPELINE_LOG = BASE / "result" / "pipeline_output.log"
MONITOR_LOG = BASE / "result" / "gpu_monitor.log"
INTERVAL = 5            # 采样间隔(秒)
MEM_WARN = 19.0         # 24GB 物理内存 80% 告警阈值(GB)
IT_DROP = 0.7           # It/sec 低于基线 70% 判定降频
TEMP_WARN = 85.0        # GPU 温度告警(°C)
TEMP_INTERVAL = 30      # 温度采样间隔(秒)
SUDO_FAIL_LIMIT = 3     # sudo 连续失败次数，超过则放弃温度读取

def get_train_proc():
    """返回训练进程 (pid, cpu%, rss_bytes) 或 (None,None,None)"""
    try:
        r = subprocess.run(["pgrep", "-f", "train_lora.py"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            pid = r.stdout.split()[0]
            r = subprocess.run(["ps", "-o", "%cpu,rss", "-p", pid], capture_output=True, text=True)
            parts = r.stdout.split("\n")[1].split()
            return pid, float(parts[0]), int(parts[1])
    except Exception:
        pass
    return None, None, None

def latest_train_stats():
    """从 pipeline log 提取最新训练行 (iter, it_sec, peak_gb)"""
    try:
        for line in reversed(PIPELINE_LOG.read_text(errors="ignore").splitlines()):
            if "Iter " in line and "Train loss" in line:
                it = re.search(r"Iter (\d+):", line)
                isec = re.search(r"It/sec ([\d.]+)", line)
                pk = re.search(r"Peak mem ([\d.]+) GB", line)
                if it and isec and pk:
                    return int(it.group(1)), float(isec.group(1)), float(pk.group(1))
    except Exception:
        pass
    return None, None, None

def get_temp():
    """GPU die temperature(°C)，需 root，失败返回 None"""
    try:
        r = subprocess.run(
            ["sudo", "-n", "powermetrics", "--samplers", "gpu_power", "-n", "1", "-i", "1000"],
            capture_output=True, text=True, timeout=15)
        if r.returncode == 0:
            m = re.search(r"GPU die temperature:\s*([\d.]+)\s*C", r.stdout)
            if m:
                return float(m.group(1))
    except Exception:
        pass
    return None

def mem_free_pct():
    try:
        r = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True)
        m = re.search(r"free percentage:\s*(\d+)%", r.stdout)
        return int(m.group(1)) if m else None
    except Exception:
        return None

def main():
    os.makedirs(MONITOR_LOG.parent, exist_ok=True)
    f = open(MONITOR_LOG, "a", buffering=1)
    base_it, base_set = 0.0, False
    last_temp, sudo_fails, temp_on = 0.0, 0, False
    print(f"MLX 监控启动 {datetime.now():%F %T} | 每{INTERVAL}s 采样 | 日志: {MONITOR_LOG}")
    f.write(f"=== MLX monitor start {datetime.now():%F %T} ===\n")

    while True:
        ts = datetime.now().strftime("%F %T")
        pid, cpu, rss = get_train_proc()
        it, it_sec, peak = latest_train_stats()
        mfree = mem_free_pct()

        # 温度：每 TEMP_INTERVAL 秒尝试，连续失败 SUDO_FAIL_LIMIT 次后放弃
        if sudo_fails < SUDO_FAIL_LIMIT and (temp_on or time.time() - last_temp >= TEMP_INTERVAL):
            temp = get_temp()
            last_temp = time.time()
            if temp is not None:
                temp_on, sudo_fails = True, 0
            else:
                sudo_fails += 1
                if sudo_fails >= SUDO_FAIL_LIMIT:
                    print("温度读取不可用（需 root/sudo），已降级为 It/sec 降频信号监控", flush=True)

        # 建立平滑滚动 It/sec 基线
        if it_sec is not None:
            base_it = it_sec if not base_set else 0.8 * base_it + 0.2 * it_sec
            base_set = True

        # 告警判断
        warns = []
        if peak is not None and peak > MEM_WARN:
            warns.append(f"显存>{MEM_WARN}GB预警!")
        if it_sec is not None and base_set and base_it > 0 and it_sec < base_it * IT_DROP:
            warns.append(f"It/sec掉{int((1-it_sec/base_it)*100)}%疑似降频!")
        if temp_on and temp is not None and temp > TEMP_WARN:
            warns.append(f"GPU{temp:.0f}°C过热!")

        proc_s = f"train pid={pid or 'N/A'} cpu={cpu if cpu is not None else 'N/A'}% rss={rss/1024/1024:.0f}MB" if pid else "train: N/A (未运行)"
        mem_s = f"peak={peak:.2f}GB@iter{it} it/s={it_sec}" if peak else "peak=N/A"
        temp_s = f"{temp:.1f}°C" if temp_on and temp is not None else "N/A(需root)"
        line = (f"{ts} | {proc_s} | {mem_s} | mem_free={mfree if mfree is not None else 'N/A'}% "
                f"| gpu_temp={temp_s}" + (f" | {' '.join(warns)}" if warns else ""))
        print(line, flush=True)
        f.write(line + "\n")
        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()
