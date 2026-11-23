"""Deterministic analytics over tenant-scoped series. No LLM, no I/O."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .data import Snapshot

DAY = 86400.0


@dataclass(frozen=True)
class Thresholds:
    disk_low_pct: float = 10.0          # "low on disk" = <10% free
    disk_critical_pct: float = 5.0
    disk_decline_pp_day: float = 0.25   # steady decline considered meaningful
    ram_high_pct: float = 85.0
    ram_consistent_share: float = 0.70  # >=70% of snapshots above ram_high_pct
    ram_intermittent_share: float = 0.25
    battery_min_cap_drop_pct: float = 1.0
    replacement_battery_days: int = 7


def ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if n < 3 or sxx == 0:
        return 0.0, my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (my + slope * (x - mx))) ** 2 for x, y in zip(xs, ys))
    return slope, my - slope * mx, (1 - ss_res / ss_tot) if ss_tot else 1.0


def disk_trend(series: list[Snapshot], th: Thresholds = Thresholds()) -> dict:
    t0 = series[0].ts
    xs = [(s.ts - t0).total_seconds() / DAY for s in series]
    ys = [s.disk_free_pct for s in series]
    slope, _, r2 = ols(xs, ys)
    latest = ys[-1]
    floor = min(ys)
    plateau = 0
    for y in reversed(ys):  # trailing snapshots stuck at the minimum
        if abs(y - floor) <= 0.05 and latest - floor <= 0.05:
            plateau += 1
        else:
            break
    declining = slope <= -th.disk_decline_pp_day and r2 >= 0.8
    if plateau >= 3 and latest <= th.disk_critical_pct:
        days_left = 0.0                      # volume effectively full; forecast meaningless
    elif declining:
        days_left = max(latest, 0.0) / -slope
    else:
        days_left = None
    return {
        "n": len(series), "first_free_pct": round(ys[0], 1), "latest_free_pct": round(latest, 1),
        "slope_pp_per_day": round(slope, 3), "r2": round(r2, 3), "declining": declining,
        "plateau_snapshots": plateau, "days_to_full": None if days_left is None else round(days_left, 1),
        "latest_free_gb": round(series[-1].disk_avail / 2**30, 1), "total_gb": round(series[-1].disk_total / 2**30),
        "snaps_below_low": sum(1 for y in ys if y < th.disk_low_pct),
    }


def ram_pressure(series: list[Snapshot], th: Thresholds = Thresholds()) -> dict:
    ys = sorted(s.ram_used_pct for s in series)
    n = len(ys)
    above = sum(1 for y in ys if y > th.ram_high_pct)
    share = above / n
    tier = ("consistent" if share >= th.ram_consistent_share
            else "intermittent" if share >= th.ram_intermittent_share else "none")
    return {"n": n, "mean_pct": round(sum(ys) / n, 1), "p95_pct": round(ys[min(n - 1, math.ceil(0.95 * n) - 1)], 1),
            "max_pct": round(ys[-1], 1), "snaps_above": above, "share_above": round(share, 2),
            "tier": tier, "ram_gb": round(series[-1].ram_total / 2**30)}


def battery_wear(series: list[Snapshot], as_of, th: Thresholds = Thresholds()) -> dict:
    reads = [s for s in series if s.battery and s.battery.get("full_charge_capacity")]
    out = {"n": len(series), "n_readings": len(reads), "n_missing": len(series) - len(reads)}
    if not reads:
        return {**out, "has_battery_data": False}
    f, l = reads[0], reads[-1]
    c0, c1 = f.battery["full_charge_capacity"], l.battery["full_charge_capacity"]
    days = max((l.ts - f.ts).total_seconds() / DAY, 1e-9)
    cond_series = [(s.ts, s.battery.get("condition")) for s in reads]
    service_since = None
    for i, (ts, c) in enumerate(cond_series):
        if c == "Service Recommended" and all(cc == "Service Recommended" for _, cc in cond_series[i:]):
            service_since = ts
            break
    return {
        **out, "has_battery_data": True, "first_capacity": c0, "last_capacity": c1,
        "capacity_change_pct": round(100.0 * (c1 - c0) / c0, 2),
        "capacity_change_pct_per_day": round(100.0 * (c1 - c0) / c0 / days, 3),
        "cycles_first": f.battery.get("cycle_count"), "cycles_last": l.battery.get("cycle_count"),
        "condition_latest_known": l.battery.get("condition"),
        "service_since": service_since.strftime("%Y-%m-%d") if service_since else None,
        "service_days": int((as_of - service_since).total_seconds() // DAY) if service_since else 0,
        "last_reading_ts": l.ts_str, "stale_days": int((as_of - l.ts).total_seconds() // DAY),
        "declining": (c1 - c0) / c0 * 100 <= -th.battery_min_cap_drop_pct,
        # NOTE: design capacity is NOT in telemetry -> absolute health % cannot be computed.
    }


def check_history(series: list[Snapshot], check_id: str) -> dict:
    st = [(s.ts_str, s.checks[check_id]["status"]) for s in series if check_id in s.checks]
    if not st:
        return {"n": 0}
    fails = [t for t, v in st if v == "fail"]
    regressions = [st[i][0] for i in range(1, len(st)) if st[i - 1][1] == "pass" and st[i][1] == "fail"]
    streak = 0
    for _, v in reversed(st):
        if v != "fail":
            break
        streak += 1
    return {"n": len(st), "latest": st[-1][1], "fail_snaps": len(fails), "first_fail_ts": fails[0] if fails else None,
            "regression_ts": regressions[-1] if regressions else None, "current_fail_streak": streak,
            "always_failing": len(fails) == len(st)}


def fleet_check_drift(view, check_id: str, windows: int = 4, width_days: int = 7) -> list[dict]:
    """Devices failing `check_id` per trailing window (last snapshot per device in window).
    Window i covers ages [i*W, (i+1)*W) days before the tenant's as_of."""
    as_of, out = view.as_of, []
    for i in range(windows - 1, -1, -1):
        failing, seen = [], 0
        for d in view.device_ids():
            snaps = [s for s in view.series(d)
                     if i * width_days * DAY <= (as_of - s.ts).total_seconds() < (i + 1) * width_days * DAY]
            if snaps and check_id in snaps[-1].checks:
                seen += 1
                if snaps[-1].checks[check_id]["status"] == "fail":
                    failing.append(d)
        out.append({"window_end_days_ago": i * width_days, "devices_seen": seen, "failing": len(failing), "failing_ids": failing})
    return out
