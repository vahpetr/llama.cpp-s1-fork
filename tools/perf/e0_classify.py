#!/usr/bin/env python3
"""E0 kernel-level classifier: compare the per-case output bytes of a variant run against the base run.
usage: e0_classify.py <base_dir> <variant_dir> [<variant_name>]   (each dir: <case>.bin + run.log with [e0-case] and
[vk-dispatch] lines). Prints one TSV row per case: case, op, type, m(used), n, class (identical / differs),
n_diff, max_abs_diff, first_diff (index: base -> variant), base kernel, variant kernel, repeat_ok."""
import os, re, struct, sys

def kernels(log):
    k = {}
    for line in open(log, errors="replace"):
        if line.startswith("[vk-dispatch]"):
            node = re.search(r"node=(\S+)", line); pipe = re.search(r"pipeline=(\S+)", line)
            path = re.search(r"path=(\S+)", line); mmq = re.search(r"mmq=(\d)", line); mmvq = re.search(r"mmvq=(\d)", line)
            if node and pipe:
                k.setdefault(node.group(1), f"{path.group(1)}:{pipe.group(1)}:mmq{mmq.group(1)}:mmvq{mmvq.group(1)}")
    return k

def cases(log):
    c = {}
    for line in open(log, errors="replace"):
        if line.startswith("[e0-case]") and "ERROR" not in line:
            d = dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv); c[d["id"]] = d
    return c

def main():
    base, var = sys.argv[1], sys.argv[2]; name = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(var.rstrip("/"))
    kb, kv = kernels(f"{base}/run.log"), kernels(f"{var}/run.log"); cb, cv = cases(f"{base}/run.log"), cases(f"{var}/run.log")
    print("variant\tcase\top\ttype\tm\tn\tclass\tn_diff\tmax_abs_diff\tfirst_diff\tkernel_base\tkernel_variant\trepeat_ok")
    for cid in sorted(cb):
        d = cb[cid]; fb, fv = f"{base}/{cid}.bin", f"{var}/{cid}.bin"
        if cid not in cv or not os.path.exists(fv):
            print(f"{name}\t{cid}\t{d['op']}\t{d['type']}\t{d['m']}\t{d['n']}\tmissing\t-\t-\t-\t{kb.get(cid,'?')}\t-\t-"); continue
        a, b = open(fb, "rb").read(), open(fv, "rb").read()
        if a == b:
            cls, nd, mx, first = "identical", 0, 0.0, "-"
        else:
            n = min(len(a), len(b)) // 4; fa = struct.unpack(f"<{n}f", a[:4*n]); fv_ = struct.unpack(f"<{n}f", b[:4*n])
            nd, mx, first = 0, 0.0, None
            for i, (x, y) in enumerate(zip(fa, fv_)):
                if struct.pack("<f", x) != struct.pack("<f", y):
                    nd += 1; dd = abs(x - y)
                    if dd > mx: mx = dd
                    if first is None: first = f"{i}: {x!r} -> {y!r}"
            cls = "differs"
        rep = f"{cb[cid].get('repeat_ok','-')}/{cv[cid].get('repeat_ok','-')}"
        print(f"{name}\t{cid}\t{d['op']}\t{d['type']}\t{d['m']}\t{d['n']}\t{cls}\t{nd}\t{mx:.6g}\t{first}\t{kb.get(cid,'?')}\t{kv.get(cid,'?')}\t{rep}")

if __name__ == "__main__":
    main()
