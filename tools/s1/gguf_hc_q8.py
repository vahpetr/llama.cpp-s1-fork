#!/usr/bin/env python3
# tools/s1/gguf_hc_q8.py - copied from my private agent repo (2026-10-10).
# Used to build the balkon hands weights (deploy/README.md):
#   1) python3 gguf_hc_q8.py IN-00001.gguf OUT-hc8-00001.gguf
#        re-encodes the BF16 hyper-connection weights (hc_*) to Q8_0 - the Radeon 780M has no bf16 math.
#   2) python3 gguf_hc_q8.py IN-hc8-00001.gguf OUT-hc8-o8-00001.gguf --from SRC.gguf --tensors output.weight,token_embd.weight
#        takes the LM head and the token embedding byte-for-byte from SRC (a Q8_0 made from BF16) instead of the low-bit copy.
# Needs numpy only. Works on one shard of a split GGUF; shards it does not touch are reused unchanged (hardlink/copy them).
# Not exact: hc8 changes the hc_* numerics (BF16 -> Q8_0); o8 swaps in higher-precision head/embedding bytes.
"""gguf_hc_q8.py IN OUT — copy a GGUF (split shard ok), re-encoding only the BF16 2-D hyper-connection weights
(names containing "hc_" and ending in ".weight", type BF16, ne0 % 32 == 0) to Q8_0 (ggml reference rounding).
KV section is copied byte-for-byte; every other tensor is copied byte-for-byte; offsets are recomputed.
Optional: --from SRC.gguf --tensors a.weight,b.weight — take those tensors (same dims, SRC's type, e.g. Q8_0
made from BF16) byte-for-byte from SRC instead (head/embedding upgrade); without it the behaviour is unchanged."""
import sys, struct, numpy as np
IN, OUT = sys.argv[1], sys.argv[2]
SRC = sys.argv[sys.argv.index("--from") + 1] if "--from" in sys.argv else None
TAKE = set(x.encode() for x in sys.argv[sys.argv.index("--tensors") + 1].split(",")) if SRC else set()
f = open(IN, "rb"); hdr = f.read(64 << 20); o = 0
def r(fmt):
    global o; v = struct.unpack_from(fmt, hdr, o); o += struct.calcsize(fmt); return v[0]
def s():
    global o; n = r("<Q"); v = hdr[o:o+n]; o += n; return v
def skip_val(t):
    if t == 8: s(); return None
    if t == 9:
        et = r("<I"); n = r("<Q")
        for _ in range(n): skip_val(et)
        return None
    return r({0:"<B",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",7:"<?",10:"<Q",11:"<q",12:"<d"}[t])
assert hdr[:4] == b"GGUF"; o = 4
ver = r("<I"); nt = r("<Q"); nk = r("<Q"); kv0 = o; align = 32
for _ in range(nk):
    k = s(); t = r("<I"); v = skip_val(t)
    if k == b"general.alignment": align = v
kv1 = o
T = []
for _ in range(nt):
    n = s(); nd = r("<I"); dims = [r("<Q") for _ in range(nd)]; t = r("<I"); off = r("<Q"); T.append([n, dims, t, off])
ds = (o + align - 1) // align * align
f.seek(0, 2); fend = f.seek(0, 2)
byoff = sorted(range(nt), key=lambda i: T[i][3])
size = {}
for j, i in enumerate(byoff):
    nxt = T[byoff[j+1]][3] if j + 1 < nt else fend - ds
    size[i] = nxt - T[i][3]
BF16, Q8_0 = 30, 8
REPL = {}                         # name -> (type, bytes) taken from SRC
if SRC:
    sf = open(SRC, "rb"); sh = sf.read(64 << 20); so = [4]
    def sr(fmt):
        v = struct.unpack_from(fmt, sh, so[0]); so[0] += struct.calcsize(fmt); return v[0]
    def ss():
        n = sr("<Q"); v = sh[so[0]:so[0]+n]; so[0] += n; return v
    def sskip(t):
        if t == 8: ss(); return None
        if t == 9:
            et = sr("<I"); n = sr("<Q")
            for _ in range(n): sskip(et)
            return None
        return sr({0:"<B",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",7:"<?",10:"<Q",11:"<q",12:"<d"}[t])
    sr("<I"); snt = sr("<Q"); snk = sr("<Q"); salign = 32
    for _ in range(snk):
        k = ss(); t = sr("<I"); v = sskip(t)
        if k == b"general.alignment": salign = v
    ST = []
    for _ in range(snt):
        n = ss(); nd = sr("<I"); dims = [sr("<Q") for _ in range(nd)]; t = sr("<I"); off = sr("<Q"); ST.append((n, dims, t, off))
    sds = (so[0] + salign - 1) // salign * salign; sf.seek(0, 2); send = sf.tell()
    soffs = sorted(x[3] for x in ST) + [send - sds]
    for n, dims, t, off in ST:
        if n in TAKE:
            mine = [x for x in T if x[0] == n]; assert mine and mine[0][1] == dims, (n, dims)
            ne = int(np.prod(dims)); nb = ne // 32 * 34 if t == Q8_0 else soffs[soffs.index(off) + 1] - off; sf.seek(sds + off); REPL[n] = (t, sf.read(nb))
    assert set(REPL) == TAKE, ("missing in SRC", TAKE - set(REPL))
def conv(i):
    n, dims, t, _ = T[i]
    return t == BF16 and b"hc_" in n and n.endswith(b".weight") and len(dims) == 2 and dims[0] % 32 == 0
def q8(raw, ne):
    x = (np.frombuffer(raw[:ne*2], dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(-1, 32)
    amax = np.abs(x).max(axis=1); d = amax / 127.0
    idd = np.where(d > 0, 1.0 / np.where(d > 0, d, 1), 0.0).astype(np.float32)
    q = np.round(x * idd[:, None]).astype(np.int8)
    out = np.empty(x.shape[0], dtype=[("d", "<f2"), ("q", "i1", 32)]); out["d"] = d.astype(np.float16); out["q"] = q
    return out.tobytes()
pad = lambda n: (align - n % align) % align
# new tensor infos
newsz = {}; newoff = {}; cur = 0
for i in range(nt):               # data laid out in original tensor-info order (same as llama.cpp writers)
    pass
order = byoff                     # keep the original data order
for i in order:
    ne = 1
    for d in T[i][1]: ne *= d
    nb = ne // 32 * 34 if conv(i) else size[i]
    if T[i][0] in REPL: nb = len(REPL[T[i][0]][1])
    newsz[i] = nb; newoff[i] = cur; cur += nb + pad(nb)
w = open(OUT, "wb")
w.write(b"GGUF" + struct.pack("<IQQ", ver, nt, nk)); w.write(hdr[kv0:kv1])
for i in range(nt):
    n, dims, t, _ = T[i]
    w.write(struct.pack("<Q", len(n)) + n + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims))
    w.write(struct.pack("<IQ", REPL[n][0] if n in REPL else Q8_0 if conv(i) else t, newoff[i]))
w.write(b"\0" * pad(w.tell()))
nconv = 0; saved = 0
for i in order:
    f.seek(ds + T[i][3]); raw = f.read(size[i])
    if T[i][0] in REPL:
        b = REPL[T[i][0]][1]; saved += size[i] - len(b)
    elif conv(i):
        ne = T[i][1][0] * T[i][1][1]; b = q8(raw, ne); nconv += 1; saved += size[i] - len(b)
    else:
        b = raw
    assert len(b) == newsz[i], (T[i][0], len(b), newsz[i])
    w.write(b); w.write(b"\0" * pad(len(b)))
w.close()
print(f"converted {nconv} tensors to Q8_0, replaced {len(REPL)} from {SRC}, saved {saved/2**30:.3f} GiB, out {OUT}")
