"""Ground truth, computed with deliberately naive code that shares NOTHING with fleet_copilot
(no imports from it, different algorithms). If both agree, a shared bug is unlikely."""
import json
from collections import defaultdict

LAPTOP_WORDS = ("macbook", "thinkpad", "xps")


def load(path):
    with open(path) as source:
        rows = [json.loads(line) for line in source if line.strip()]
    by = defaultdict(list)
    for r in rows:
        by[r["device_id"]].append(r)
    for v in by.values():
        v.sort(key=lambda r: r["collected_at"])
    return rows, by


def free_pct(r):
    return 100 * sum(v["available_bytes"] for v in r["disk_volumes"]) / sum(v["size_bytes"] for v in r["disk_volumes"])


class Truth:
    def __init__(self, path):
        self.rows, self.by = load(path)

    def devs(self, tenant):
        return sorted(d for d, v in self.by.items() if v[0]["company_id"] == tenant)

    def low_disk(self, tenant, thr=10.0):
        return {d for d in self.devs(tenant) if free_pct(self.by[d][-1]) < thr}

    def failing(self, tenant, severity=None, check=None, laptops_only=False):
        out = set()
        for d in self.devs(tenant):
            last = self.by[d][-1]
            if laptops_only and not any(w in last["device_identity"]["model_name"].lower() for w in LAPTOP_WORDS):
                continue
            for c in last["compliance_results"]:
                if c["status"] == "fail" and (not severity or c["severity"] == severity) and (not check or c["check_id"] == check):
                    out.add(d)
        return out

    def os_older(self, tenant, major):
        return {d for d in self.devs(tenant)
                if self.by[d][-1]["os"]["product_name"] == "macOS" and int(self.by[d][-1]["os"]["product_version"].split(".")[0]) < major}

    def ram_consistent(self, tenant):
        out = set()
        for d in self.devs(tenant):
            v = self.by[d]
            hi = sum(1 for r in v if r["memory"]["used_memory_bytes"] / r["memory"]["total_memory_bytes"] > 0.85)
            if hi / len(v) >= 0.7:
                out.add(d)
        return out

    def battery_service(self, tenant):
        out = set()
        for d in self.devs(tenant):
            conds = [r["battery"]["condition"] for r in self.by[d] if r.get("battery")]
            if conds and conds[-1] == "Service Recommended":
                out.add(d)
        return out

    def disk_critical(self, tenant):
        return {d for d in self.devs(tenant) if free_pct(self.by[d][-1]) <= 5.0}

    def screen_lock_regressions(self, tenant):
        out = set()
        for d in self.devs(tenant):
            st = [next(c["status"] for c in r["compliance_results"] if c["check_id"] == "screen_lock") for r in self.by[d]]
            if st[-1] == "fail" and "pass" in st[:-1] and any(st[i - 1] == "pass" and st[i] == "fail" for i in range(1, len(st))):
                out.add(d)
        return out

    def employee_of(self, d):
        return self.by[d][-1]["employee_id"]

    def raw(self, device_id, ts):
        return next((r for r in self.by.get(device_id, []) if r["collected_at"] == ts), None)
