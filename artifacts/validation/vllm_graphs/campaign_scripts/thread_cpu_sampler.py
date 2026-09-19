"""Sample per-thread CPU time from /proc for the driver + EngineCore processes.

No ptrace needed. Waits for a GPU compute process to appear, then samples
utime+stime per thread (and process-level) every INTERVAL seconds until the
target exits. Output: CSV rows  ts,pid,role,tid,comm,cpu_ticks_delta
"""
import os
import subprocess
import sys
import time

OUT = sys.argv[1]
INTERVAL = 2.0
HZ = os.sysconf("SC_CLK_TCK")


def gpu_pids():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        return []
    return sorted({int(x) for x in out.split() if x.strip().isdigit()})


def ppid_of(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            parts = f.read().rsplit(")", 1)[1].split()
        return int(parts[1])
    except Exception:
        return None


def thread_cpu(pid):
    res = {}
    try:
        for tid in os.listdir(f"/proc/{pid}/task"):
            try:
                with open(f"/proc/{pid}/task/{tid}/stat") as f:
                    raw = f.read()
                comm = raw[raw.index("(") + 1 : raw.rindex(")")]
                parts = raw.rsplit(")", 1)[1].split()
                utime, stime = int(parts[11]), int(parts[12])
                res[int(tid)] = (comm, utime + stime)
            except Exception:
                continue
    except FileNotFoundError:
        return None
    return res


def main():
    deadline = time.time() + 600
    pids = []
    while time.time() < deadline:
        pids = gpu_pids()
        if pids:
            break
        time.sleep(2)
    if not pids:
        print("no gpu process appeared", file=sys.stderr)
        return
    eng = pids[-1]
    drv = ppid_of(eng)
    roles = {eng: "engine"}
    if drv and drv > 1:
        roles[drv] = "driver"
    prev = {}
    with open(OUT, "w") as f:
        f.write("ts,pid,role,tid,comm,cpu_ticks_delta\n")
        while True:
            ts = time.time()
            alive = False
            for pid, role in roles.items():
                snap = thread_cpu(pid)
                if snap is None:
                    continue
                alive = True
                for tid, (comm, total) in snap.items():
                    d = total - prev.get((pid, tid), total)
                    prev[(pid, tid)] = total
                    if d:
                        f.write(f"{ts:.1f},{pid},{role},{tid},{comm},{d}\n")
            f.flush()
            if not alive:
                break
            time.sleep(INTERVAL)
    print(f"sampler done (engine={eng} driver={drv}, HZ={HZ})", file=sys.stderr)


if __name__ == "__main__":
    main()
