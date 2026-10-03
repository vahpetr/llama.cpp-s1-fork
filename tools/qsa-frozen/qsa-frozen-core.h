// [TAG_QSA_FROZEN] llama-qsa-frozen core: the capture format, the capture eval callback and the frozen-tensor replay
// (frozen-M6 isolation, ~/claude-tmp/codex/m6-frozen.sh). Header-only, ggml + ggml-backend + nlohmann::json only, so
// tests/test-qsa-frozen.cpp drives exactly the code the tool runs.
//
// Capture layout, one directory per (depth, layer): <capture-dir>/cap-<d>-il<il>/
//   q.bin      f32  every chunk's flash-attention src[0] as fed (post-rope, post-Hadamard), made contiguous in its
//                   logical [D_k, nc, n_head] order, chunks in graph order (so row t of the ubatch is query t)
//   k.bin      raw  src[1]'s cache bytes of cells [0, n_kv): the cache layout (cell-major, n_head_kv rows of D_k per
//                   cell, the cache's type, e.g. q8_0), written once per layer; meta.k = type/ne/nb of the view
//   v.bin      raw  the same for src[2]
//   ids.bin    i32  [width, n_tps]: indexer_top_k-<il>, chunks concatenated
//   kv_ids.bin i32  [width, n_tps]: src[5] of the indexed flash attention, when the fill ran indexed (else absent)
//   cells.bin  i32  [n_kv]: the attention cache's position per cell, -1 = empty
//   toks.bin   i32  [n_tps]: the probe ubatch's token positions
//   mask.fnv   text "<t0> <nc> <fnv1a-64 hex>" per chunk: FNV-1a of src[3] (the f16 mask flash attention read)
//   meta.json  depth, il, n_kv, n_tps, width, chunk, chunks[{t0,nc}], d_k, d_v, n_head, n_head_kv, types / ne / nb of
//              q k v, kq_scale, max_bias, softcap, prec, hadamard, qsa_prefill, git_rev
#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace qf {

using json = nlohmann::ordered_json;

// ---------------------------------------------------------------------------------------------------------------- utils

static inline uint64_t fnv1a(const void * data, size_t n, uint64_t h = 0xcbf29ce484222325ull) {
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

static inline std::string hex64(uint64_t v) {
    char b[24];
    snprintf(b, sizeof(b), "%016" PRIx64, v);
    return b;
}

// "65536,212992" -> {65536, 212992}; every item a non-negative integer, at least one
static inline std::vector<int64_t> parse_list(const std::string & s) {
    std::vector<int64_t> out;
    size_t i = 0;
    while (i <= s.size()) {
        const size_t j = std::min(s.find(',', i), s.size());
        const std::string item = s.substr(i, j - i);
        if (item.empty() || item.find_first_not_of("0123456789") != std::string::npos || item.size() > 12) {
            throw std::invalid_argument("not a list of non-negative integers: '" + s + "'");
        }
        out.push_back(std::stoll(item));
        i = j + 1;
    }
    return out;
}

static inline void write_file(const std::string & path, const void * data, size_t n, bool append = false) {
    FILE * f = fopen(path.c_str(), append ? "ab" : "wb");
    if (!f) {
        throw std::runtime_error("cannot write " + path);
    }
    if (n > 0 && fwrite(data, 1, n, f) != n) {
        fclose(f);
        throw std::runtime_error("short write " + path);
    }
    fclose(f);
}

static inline std::vector<uint8_t> read_file(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("cannot read " + path);
    }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf((size_t) std::max<long>(n, 0));
    if (n > 0 && fread(buf.data(), 1, (size_t) n, f) != (size_t) n) {
        fclose(f);
        throw std::runtime_error("short read " + path);
    }
    fclose(f);
    return buf;
}

static inline bool file_exists(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (f) {
        fclose(f);
    }
    return f != nullptr;
}

// a tensor's data in its LOGICAL order, rows contiguous (works for views and permuted views on any backend: reads
// the tensor's byte extent once, then gathers the rows with its strides). Rows must be contiguous (nb[0] = type size).
static inline std::vector<uint8_t> tensor_bytes(const ggml_tensor * t) {
    const size_t row = ggml_row_size(t->type, t->ne[0]);
    std::vector<uint8_t> out(row*t->ne[1]*t->ne[2]*t->ne[3]);
    if (ggml_is_contiguous(t)) {
        ggml_backend_tensor_get(t, out.data(), 0, out.size());
        return out;
    }
    if (t->nb[0] != ggml_type_size(t->type)) {
        throw std::runtime_error(std::string("tensor_bytes: rows of ") + t->name + " are not contiguous");
    }
    std::vector<uint8_t> ext(ggml_nbytes(t));
    ggml_backend_tensor_get(t, ext.data(), 0, ext.size());
    size_t o = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                memcpy(out.data() + o, ext.data() + i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3], row);
                o += row;
            }
        }
    }
    return out;
}

static inline json tensor_desc(const ggml_tensor * t) {
    return json { { "type", ggml_type_name(t->type) },
                  { "ne", { t->ne[0], t->ne[1], t->ne[2], t->ne[3] } },
                  { "nb", { t->nb[0], t->nb[1], t->nb[2], t->nb[3] } } };
}

static inline ggml_type type_from_name(const std::string & s) {
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const char * n = ggml_type_name((ggml_type) i);
        if (n && s == n) {
            return (ggml_type) i;
        }
    }
    throw std::runtime_error("unknown ggml type " + s);
}

// the layer of a node: the first "<name>-<il>" (ignoring " (view)"-style suffixes) on the way up through view_src / src[]
static inline int layer_of(const ggml_tensor * t) {
    std::vector<const ggml_tensor *> frontier = { t };
    for (int depth = 0; depth < 8 && !frontier.empty(); ++depth) {
        std::vector<const ggml_tensor *> next;
        for (const ggml_tensor * x : frontier) {
            std::string n = x->name;
            const size_t sp = n.find(' ');
            if (sp != std::string::npos) {
                n = n.substr(0, sp);
            }
            const size_t dash = n.rfind('-');
            if (dash != std::string::npos && dash + 1 < n.size() &&
                    n.find_first_not_of("0123456789", dash + 1) == std::string::npos) {
                return std::stoi(n.substr(dash + 1));
            }
            if (x->view_src) {
                next.push_back(x->view_src);
            }
            for (int i = 0; i < 2; ++i) {
                if (x->src[i]) {
                    next.push_back(x->src[i]);
                }
            }
        }
        frontier.swap(next);
    }
    return -1;
}

// ------------------------------------------------------------------------------------------------------------ capture

// One armed probe ubatch: the eval callback asks for every FLASH_ATTN_EXT of a target layer and for indexer_top_k-<il>,
// and writes them under dir/cap-<depth>-il<il>/. Disarmed, ask() is false for every node: no extra splits.
struct capture {
    bool             armed = false;
    std::set<int>    layers;
    std::string      dir;             // <capture-dir>
    int64_t          depth = 0;
    bool             hadamard = false;
    std::string      qsa_prefill;
    std::string      git_rev;

    struct layer_state {
        std::string        path;
        int64_t            n_kv = -1, d_k = 0, d_v = 0, n_head = 0, n_head_kv = 0, width = -1;
        int64_t            n_fa = 0, n_ids = 0, t_fa = 0, t_ids = 0;
        float              scale = 0, max_bias = 0, softcap = 0;
        int                prec = 0;
        bool               kv_ids = false;
        json               q_desc, k_desc, v_desc, chunks = json::array();
        std::vector<std::string> fnv_lines;
        size_t             bytes = 0;
    };
    std::map<int, layer_state> st;

    std::string cap_dir(int il) const { return dir + "/cap-" + std::to_string(depth) + "-il" + std::to_string(il); }

    void arm(int64_t d) {
        depth = d;
        st.clear();
        for (int il : layers) {
            layer_state & s = st[il];
            s.path = cap_dir(il);
            if (system(("mkdir -p '" + s.path + "'").c_str()) != 0) {
                throw std::runtime_error("cannot create " + s.path);
            }
            for (const char * f : { "q.bin", "k.bin", "v.bin", "ids.bin", "kv_ids.bin" }) {
                remove((s.path + "/" + f).c_str());
            }
        }
        armed = true;
    }

    // is t a node to capture, and for which layer (-1 = no)
    int wants(const ggml_tensor * t) const {
        if (!armed) {
            return -1;
        }
        if (t->op == GGML_OP_FLASH_ATTN_EXT) {
            const int il = layer_of(t->src[0]);
            return layers.count(il) ? il : -1;
        }
        const char * pre = "indexer_top_k-";
        if (strncmp(t->name, pre, strlen(pre)) == 0) {
            const std::string rest = t->name + strlen(pre);
            if (!rest.empty() && rest.find_first_not_of("0123456789") == std::string::npos) {
                const int il = std::stoi(rest);
                return layers.count(il) ? il : -1;
            }
        }
        return -1;
    }

    // ggml_backend_sched_eval_callback
    bool on_eval(ggml_tensor * t, bool ask) {
        if (ask) {
            if (armed && !hadamard) {
                for (int i = 0; i < GGML_MAX_SRC; ++i) {
                    if (t->src[i] && strcmp(t->src[i]->name, "attn_inp_k_rot") == 0) {
                        hadamard = true;
                    }
                }
            }
            return wants(t) >= 0;
        }
        const int il = wants(t);
        if (il < 0) {
            return true;
        }
        layer_state & s = st[il];
        if (t->op == GGML_OP_FLASH_ATTN_EXT) {
            const ggml_tensor * q = t->src[0], * k = t->src[1], * v = t->src[2], * m = t->src[3], * ids = t->src[5];
            if (q->type != GGML_TYPE_F32 || q->ne[3] != 1 || k->ne[3] != 1) {
                throw std::runtime_error("capture: unexpected flash attention shape at il " + std::to_string(il));
            }
            const int64_t nc = q->ne[1];
            if (s.n_kv < 0) {
                s.n_kv = k->ne[1]; s.d_k = k->ne[0]; s.d_v = v->ne[0]; s.n_head = q->ne[2]; s.n_head_kv = k->ne[2];
                s.k_desc = tensor_desc(k); s.v_desc = tensor_desc(v);
                memcpy(&s.scale,    (const float *) t->op_params + 0, sizeof(float));
                memcpy(&s.max_bias, (const float *) t->op_params + 1, sizeof(float));
                memcpy(&s.softcap,  (const float *) t->op_params + 2, sizeof(float));
                s.prec = ggml_flash_attn_ext_get_prec(t);
                // the cache layout: one cell = n_head_kv rows of D_k, i.e. the extent of the view is n_kv whole cells
                for (const ggml_tensor * x : { k, v }) {
                    if (x->nb[1] != ggml_row_size(x->type, x->ne[0])*x->ne[2] || x->nb[2] != ggml_row_size(x->type, x->ne[0])) {
                        throw std::runtime_error("capture: K/V is not a cell-major cache view");
                    }
                    std::vector<uint8_t> raw(ggml_nbytes(x));
                    ggml_backend_tensor_get(x, raw.data(), 0, raw.size());
                    write_file(s.path + (x == k ? "/k.bin" : "/v.bin"), raw.data(), raw.size());
                    s.bytes += raw.size();
                }
            } else if (k->ne[1] != s.n_kv) {
                throw std::runtime_error("capture: n_kv changed inside the probe ubatch");
            }
            if (s.q_desc.is_null()) {
                s.q_desc = tensor_desc(q);
            }
            const std::vector<uint8_t> qb = tensor_bytes(q);
            write_file(s.path + "/q.bin", qb.data(), qb.size(), true);
            s.bytes += qb.size();
            uint64_t h = 0;
            if (m) {
                const std::vector<uint8_t> mb = tensor_bytes(m);
                h = fnv1a(mb.data(), mb.size());
            }
            s.fnv_lines.push_back(std::to_string(s.t_fa) + " " + std::to_string(nc) + " " + hex64(h));
            s.chunks.push_back(json { { "t0", s.t_fa }, { "nc", nc } });
            if (ids) {
                s.kv_ids = true;
                const std::vector<uint8_t> ib = tensor_bytes(ids);
                write_file(s.path + "/kv_ids.bin", ib.data(), ib.size(), true);
            }
            s.t_fa += nc;
            s.n_fa++;
        } else {
            if (t->type != GGML_TYPE_I32) {
                throw std::runtime_error("capture: indexer_top_k is not I32");
            }
            if (s.width < 0) {
                s.width = t->ne[0];
            }
            const std::vector<uint8_t> ib = tensor_bytes(t);
            write_file(s.path + "/ids.bin", ib.data(), ib.size(), true);
            s.t_ids += t->ne[1]*t->ne[2]*t->ne[3];
            s.n_ids++;
        }
        return true;
    }

    // after the probe ubatch: cells / toks / mask.fnv / meta.json per layer; returns {il: summary}
    json finish(const std::vector<int32_t> & cells, const std::vector<int32_t> & toks) {
        armed = false;
        json out = json::object();
        for (auto & [il, s] : st) {
            if (s.n_fa == 0 || s.n_ids == 0) {
                throw std::runtime_error("capture: layer " + std::to_string(il) + " produced " + std::to_string(s.n_fa) +
                        " flash-attention and " + std::to_string(s.n_ids) + " indexer_top_k nodes (is it a QSA layer above "
                        "LLAMA_QSA_MIN_KV, and the ubatch on the scan/indexed path?)");
            }
            if (s.t_fa != (int64_t) toks.size() || s.t_ids != (int64_t) toks.size()) {
                throw std::runtime_error("capture: layer " + std::to_string(il) + " covered " + std::to_string(s.t_fa) + " / " +
                        std::to_string(s.t_ids) + " queries, the ubatch has " + std::to_string(toks.size()));
            }
            if ((int64_t) cells.size() < s.n_kv) {
                throw std::runtime_error("capture: fewer cell positions than n_kv");
            }
            write_file(s.path + "/cells.bin", cells.data(), (size_t) s.n_kv*sizeof(int32_t));
            write_file(s.path + "/toks.bin", toks.data(), toks.size()*sizeof(int32_t));
            std::string fnv;
            for (const auto & l : s.fnv_lines) {
                fnv += l + "\n";
            }
            write_file(s.path + "/mask.fnv", fnv.data(), fnv.size());
            int64_t chunk = 0;
            for (const auto & c : s.chunks) {
                chunk = std::max<int64_t>(chunk, c["nc"].get<int64_t>());
            }
            json meta = {
                { "depth", depth }, { "il", il }, { "n_kv", s.n_kv }, { "n_tps", (int64_t) toks.size() }, { "width", s.width },
                { "chunk", chunk }, { "chunks", s.chunks },
                { "d_k", s.d_k }, { "d_v", s.d_v }, { "n_head", s.n_head }, { "n_head_kv", s.n_head_kv },
                { "q", s.q_desc }, { "k", s.k_desc }, { "v", s.v_desc },
                { "q_file_layout", "f32 [d_k, nc, n_head] per chunk, chunks in order" },
                { "kv_file_layout", "the cache's cells [0, n_kv): n_head_kv rows of d_k per cell" },
                { "kq_scale", s.scale }, { "max_bias", s.max_bias }, { "softcap", s.softcap }, { "prec", s.prec },
                { "hadamard", hadamard }, { "kv_ids", s.kv_ids }, { "qsa_prefill", qsa_prefill }, { "git_rev", git_rev },
            };
            const std::string ms = meta.dump(1) + "\n";
            write_file(s.path + "/meta.json", ms.data(), ms.size());
            out[std::to_string(il)] = json { { "dir", s.path }, { "fa_nodes", s.n_fa }, { "bytes", s.bytes } };
        }
        st.clear();
        return out;
    }
};

static inline bool capture_eval_cb(ggml_tensor * t, bool ask, void * ud) {
    return ((capture *) ud)->on_eval(t, ask);
}

// ------------------------------------------------------------------------------------------------------------- replay

struct pair_stats {
    double  max_abs = 0, num = 0, den = 0;
    int64_t rows_gt = 0, argmax_row = -1;
};

struct replay_result {
    json membership;
    std::map<std::string, pair_stats> pairs;
    json info = json::object();
    std::string error;
};

// FA mask values as the graph writes them (f16): 0 or -inf
static inline ggml_fp16_t f16_zero()  { return ggml_fp32_to_fp16(0.0f); }
static inline ggml_fp16_t f16_ninf()  { return ggml_fp32_to_fp16(-INFINITY); }

// a = test, b = reference; both [d_v, n_head, n_q] in FA output order; rows are (q, h)
static inline void accumulate(pair_stats & p, const float * a, const double * bd, const float * bf, int64_t d_v, int64_t n_rows,
                              int64_t row0) {
    for (int64_t r = 0; r < n_rows; ++r) {
        double rmax = 0;
        for (int64_t d = 0; d < d_v; ++d) {
            const double x = a[r*d_v + d];
            const double y = bd ? bd[r*d_v + d] : (double) bf[r*d_v + d];
            const double e = std::fabs(x - y);
            p.num += (x - y)*(x - y);
            p.den += y*y;
            rmax = std::max(rmax, e);
            if (e > p.max_abs || p.argmax_row < 0) {
                if (e > p.max_abs) {
                    p.max_abs = e;
                }
                p.argmax_row = row0 + r;
            }
        }
        p.rows_gt += rmax > 1e-3 ? 1 : 0;
    }
}

// Replay one capture directory: S (scan mask), I (indexed, captured ids), Is (indexed, sorted ids) on `backend`, O on the
// host in f64. Membership first; the outputs are compared only when the captured ids pass it.
static inline replay_result replay(const std::string & capdir, ggml_backend_t backend, int n_threads, bool oracle,
                                   int64_t chunk_check = 0) {
    replay_result R;
    const std::vector<uint8_t> meta_raw = read_file(capdir + "/meta.json");
    const json meta = json::parse(std::string((const char *) meta_raw.data(), meta_raw.size()));
    const int64_t n_kv = meta["n_kv"], n_tps = meta["n_tps"], W = meta["width"];
    const int64_t d_k = meta["d_k"], d_v = meta["d_v"], n_head = meta["n_head"], n_head_kv = meta["n_head_kv"];
    const ggml_type tk = type_from_name(meta["k"]["type"]), tv = type_from_name(meta["v"]["type"]);
    const float scale = meta["kq_scale"], max_bias = meta["max_bias"], softcap = meta["softcap"];
    const int prec = meta["prec"];
    const std::string mode = meta.value("qsa_prefill", "");
    std::vector<std::pair<int64_t, int64_t>> chunks;
    for (const auto & c : meta["chunks"]) {
        chunks.emplace_back(c["t0"].get<int64_t>(), c["nc"].get<int64_t>());
    }
    R.info = json { { "depth", meta["depth"] }, { "il", meta["il"] }, { "n_kv", n_kv }, { "n_tps", n_tps }, { "width", W },
                    { "chunk", meta["chunk"] }, { "n_chunks", (int64_t) chunks.size() }, { "fill_mode", mode } };
    if (chunk_check > 0 && meta["chunk"].get<int64_t>() != chunk_check) {
        R.error = "captured chunk " + std::to_string(meta["chunk"].get<int64_t>()) + " != --chunk " + std::to_string(chunk_check);
        return R;
    }
    if (max_bias != 0.0f) {
        R.error = "max_bias != 0 (ALiBi) is not modelled by the oracle";
        return R;
    }

    // ---- inputs
    const std::vector<uint8_t> qraw = read_file(capdir + "/q.bin");
    const std::vector<uint8_t> kraw = read_file(capdir + "/k.bin");
    const std::vector<uint8_t> vraw = read_file(capdir + "/v.bin");
    const std::vector<uint8_t> iraw = read_file(capdir + "/ids.bin");
    const std::vector<uint8_t> craw = read_file(capdir + "/cells.bin");
    const std::vector<uint8_t> traw = read_file(capdir + "/toks.bin");
    const size_t k_cell = ggml_row_size(tk, d_k)*n_head_kv, v_cell = ggml_row_size(tv, d_v)*n_head_kv;
    if (qraw.size() != (size_t) d_k*n_tps*n_head*4 || kraw.size() != k_cell*n_kv || vraw.size() != v_cell*n_kv ||
        iraw.size() != (size_t) W*n_tps*4 || craw.size() != (size_t) n_kv*4 || traw.size() != (size_t) n_tps*4) {
        R.error = "capture file sizes do not match meta.json";
        return R;
    }
    const float   * qall  = (const float *)   qraw.data();
    const int32_t * ids   = (const int32_t *) iraw.data();
    const int32_t * cells = (const int32_t *) craw.data();
    const int32_t * toks  = (const int32_t *) traw.data();

    auto visible = [&](int64_t t, int64_t c) { return cells[c] >= 0 && cells[c] <= toks[t]; };

    // ---- membership (before any output)
    bool unique = true, in_range = true, visible_ok = true, sorted_same = true, kv_ids_match = true;
    int64_t n_vis_min = INT64_MAX, bad_rows = 0;
    std::vector<int32_t> ids_sorted(ids, ids + W*n_tps);
    for (int64_t t = 0; t < n_tps; ++t) {
        const int32_t * row = ids + t*W;
        int32_t *       srt = ids_sorted.data() + t*W;
        std::sort(srt, srt + W);
        bool ok_u = std::adjacent_find(srt, srt + W) == srt + W;
        bool ok_r = srt[0] >= 0 && srt[W - 1] < n_kv;
        int64_t n_vis = 0, n_sel_vis = 0;
        for (int64_t c = 0; c < n_kv; ++c) {
            n_vis += visible(t, c) ? 1 : 0;
        }
        for (int64_t j = 0; j < W && ok_r; ++j) {
            n_sel_vis += visible(t, row[j]) ? 1 : 0;
        }
        const bool ok_v = ok_r && n_sel_vis == std::min<int64_t>(W, n_vis);
        std::multiset<int32_t> a(row, row + W), b(srt, srt + W);
        sorted_same &= a == b;
        unique &= ok_u; in_range &= ok_r; visible_ok &= ok_v;
        bad_rows += (ok_u && ok_r && ok_v) ? 0 : 1;
        n_vis_min = std::min(n_vis_min, n_vis);
    }
    if (file_exists(capdir + "/kv_ids.bin")) {
        const std::vector<uint8_t> kr = read_file(capdir + "/kv_ids.bin");
        kv_ids_match = kr.size() == iraw.size() && memcmp(kr.data(), iraw.data(), kr.size()) == 0;
    }

    // the regenerated FA mask of the capture's mode, per chunk, against the captured hash
    std::vector<ggml_fp16_t> m_causal, m_scan;
    auto build_masks = [&](int64_t t0, int64_t nc) {
        m_causal.assign((size_t) n_kv*nc, f16_ninf());
        m_scan  .assign((size_t) n_kv*nc, f16_ninf());
        for (int64_t i = 0; i < nc; ++i) {
            for (int64_t c = 0; c < n_kv; ++c) {
                if (visible(t0 + i, c)) {
                    m_causal[i*n_kv + c] = f16_zero();
                }
            }
            if (in_range) {
                for (int64_t j = 0; j < W; ++j) {
                    const int32_t c = ids[(t0 + i)*W + j];
                    m_scan[i*n_kv + c] = m_causal[i*n_kv + c];
                }
            }
        }
    };
    bool mask_match = true;
    {
        const std::vector<uint8_t> fr = read_file(capdir + "/mask.fnv");
        std::string txt((const char *) fr.data(), fr.size());
        size_t pos = 0;
        for (const auto & [t0, nc] : chunks) {
            const size_t e = txt.find('\n', pos);
            const std::string line = txt.substr(pos, e - pos);
            pos = e + 1;
            build_masks(t0, nc);
            const auto & m = mode == "indexed" ? m_causal : m_scan;
            const std::string h = hex64(fnv1a(m.data(), m.size()*sizeof(ggml_fp16_t)));
            mask_match &= line.size() >= 16 && line.substr(line.size() - 16) == h;
        }
    }
    R.membership = json { { "rows", n_tps }, { "unique", unique }, { "in_range", in_range }, { "visible_ok", visible_ok },
                          { "sorted_same_set", sorted_same }, { "mask_fnv_match", mask_match }, { "kv_ids_match", kv_ids_match },
                          { "n_visible_min", n_vis_min }, { "bad_rows", bad_rows } };
    if (!(unique && in_range)) {
        R.error = "membership failed (ids not unique / out of range): outputs not compared";
        return R;
    }

    // ---- K/V on the backend, once
    ggml_init_params kp = { 4*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx_kv = ggml_init(kp);
    ggml_tensor * k_base = ggml_new_tensor_2d(ctx_kv, tk, d_k*n_head_kv, n_kv);
    ggml_tensor * v_base = ggml_new_tensor_2d(ctx_kv, tv, d_v*n_head_kv, n_kv);
    ggml_backend_buffer_t buf_kv = ggml_backend_alloc_ctx_tensors(ctx_kv, backend);
    if (!buf_kv) {
        ggml_free(ctx_kv);
        R.error = "cannot allocate K/V on the backend";
        return R;
    }
    ggml_backend_tensor_set(k_base, kraw.data(), 0, kraw.size());
    ggml_backend_tensor_set(v_base, vraw.data(), 0, vraw.size());

    // ---- host f32 copies of K / V for the oracle
    std::vector<float> kf, vf;
    if (oracle) {
        kf.resize((size_t) n_kv*n_head_kv*d_k);
        vf.resize((size_t) n_kv*n_head_kv*d_v);
        const auto * trk = ggml_get_type_traits(tk);
        const auto * trv = ggml_get_type_traits(tv);
        if (!trk->to_float || !trv->to_float) {
            R.error = "no to_float for the K/V type";
            return R;
        }
        trk->to_float(kraw.data(), kf.data(), (int64_t) kf.size());
        trv->to_float(vraw.data(), vf.data(), (int64_t) vf.size());
    }

    const char * names[6] = { "I-S", "Is-S", "I-Is", "S-O", "I-O", "Is-O" };
    for (const char * n : names) {
        R.pairs[n] = pair_stats();
    }
    bool indexed_ok = true;

    for (const auto & [t0, nc] : chunks) {
        build_masks(t0, nc);

        ggml_init_params cp = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(cp);
        ggml_tensor * q   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_k, nc, n_head, 1);
        ggml_tensor * mc  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, nc, 1, 1);
        ggml_tensor * ms  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, nc, 1, 1);
        ggml_tensor * id  = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, W, nc, 1, 1);
        ggml_tensor * ids_ = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, W, nc, 1, 1);
        // the graph's views: [d, n_head_kv, n_kv] over the cell-major cache, permuted to [d, n_kv, n_head_kv]
        ggml_tensor * k = ggml_permute(ctx, ggml_view_3d(ctx, k_base, d_k, n_head_kv, n_kv, ggml_row_size(tk, d_k), k_base->nb[1], 0), 0, 2, 1, 3);
        ggml_tensor * v = ggml_permute(ctx, ggml_view_3d(ctx, v_base, d_v, n_head_kv, n_kv, ggml_row_size(tv, d_v), v_base->nb[1], 0), 0, 2, 1, 3);

        auto fa = [&](ggml_tensor * mask, ggml_tensor * kv_ids) {
            ggml_tensor * o = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, max_bias, softcap);
            ggml_flash_attn_ext_set_prec(o, (ggml_prec) prec);
            if (kv_ids) {
                ggml_flash_attn_ext_set_kv_ids(o, kv_ids);
            }
            return o;
        };
        for (ggml_tensor * x : { q, mc, ms, id, ids_ }) {
            ggml_set_input(x);
        }
        ggml_tensor * oS  = fa(ms, nullptr);
        ggml_tensor * oI  = fa(mc, id);
        ggml_tensor * oIs = fa(mc, ids_);
        for (ggml_tensor * x : { oS, oI, oIs }) {
            ggml_set_output(x);
        }
        const bool sup = ggml_backend_supports_op(backend, oI) && ggml_backend_supports_op(backend, oIs);
        if (!sup) {
            indexed_ok = false;
        }

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, oS);
        if (sup) {
            ggml_build_forward_expand(gf, oI);
            ggml_build_forward_expand(gf, oIs);
        }
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(ga, gf)) {
            ggml_gallocr_free(ga); ggml_free(ctx); ggml_backend_buffer_free(buf_kv); ggml_free(ctx_kv);
            R.error = "cannot allocate the replay graph";
            return R;
        }
        // the inputs are leaves of the graph, allocated by gallocr; q's rows of this chunk
        ggml_backend_tensor_set(q, qall + (size_t) t0*d_k*n_head, 0, ggml_nbytes(q));
        ggml_backend_tensor_set(ms, m_scan.data(), 0, ggml_nbytes(ms));
        if (sup) {
            // the indexed inputs are in the graph (and allocated) only when the backend runs the indexed FA
            ggml_backend_tensor_set(mc, m_causal.data(), 0, ggml_nbytes(mc));
            ggml_backend_tensor_set(id, ids + t0*W, 0, ggml_nbytes(id));
            ggml_backend_tensor_set(ids_, ids_sorted.data() + t0*W, 0, ggml_nbytes(ids_));
        }
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            ggml_gallocr_free(ga); ggml_free(ctx); ggml_backend_buffer_free(buf_kv); ggml_free(ctx_kv);
            R.error = "graph compute failed";
            return R;
        }
        const int64_t n_rows = nc*n_head;
        std::vector<float> S(n_rows*d_v), I(n_rows*d_v), Is(n_rows*d_v);
        ggml_backend_tensor_get(oS, S.data(), 0, S.size()*4);
        if (sup) {
            ggml_backend_tensor_get(oI, I.data(), 0, I.size()*4);
            ggml_backend_tensor_get(oIs, Is.data(), 0, Is.size()*4);
        }
        ggml_gallocr_free(ga);
        ggml_free(ctx);

        // oracle: softmax(scale q.k) v over the selected visible cells, f64; FA output order [d_v, n_head, nc]
        std::vector<double> O;
        if (oracle) {
            O.assign(n_rows*d_v, 0.0);
            const int64_t gqa = n_head/n_head_kv;
            auto work = [&](int64_t r0, int64_t r1) {
                std::vector<double> s(W), acc(d_v);
                for (int64_t r = r0; r < r1; ++r) {
                    const int64_t i = r / n_head, h = r % n_head, t = t0 + i, kh = h / gqa;
                    const float * qr = qall + ((size_t) t0*n_head*d_k) + ((size_t) h*nc + i)*d_k;
                    double mx = -INFINITY;
                    for (int64_t j = 0; j < W; ++j) {
                        const int32_t c = ids[t*W + j];
                        if (!visible(t, c)) { s[j] = -INFINITY; continue; }
                        const float * kr = kf.data() + ((size_t) c*n_head_kv + kh)*d_k;
                        double dot = 0;
                        for (int64_t d = 0; d < d_k; ++d) { dot += (double) qr[d]*(double) kr[d]; }
                        double x = (double) scale*dot;
                        if (softcap != 0.0f) { x = (double) softcap*std::tanh(x/(double) softcap); }
                        s[j] = x; mx = std::max(mx, x);
                    }
                    std::fill(acc.begin(), acc.end(), 0.0);
                    double sum = 0;
                    for (int64_t j = 0; j < W; ++j) {
                        if (s[j] == -INFINITY) { continue; }
                        const double p = std::exp(s[j] - mx);
                        sum += p;
                        const float * vr = vf.data() + ((size_t) ids[t*W + j]*n_head_kv + kh)*d_v;
                        for (int64_t d = 0; d < d_v; ++d) { acc[d] += p*(double) vr[d]; }
                    }
                    double * o = O.data() + ((size_t) i*n_head + h)*d_v;
                    for (int64_t d = 0; d < d_v; ++d) { o[d] = sum > 0 ? acc[d]/sum : 0.0; }
                }
            };
            const int nt = std::max(1, n_threads);
            std::vector<std::thread> th;
            for (int k_ = 0; k_ < nt; ++k_) {
                th.emplace_back(work, n_rows*k_/nt, n_rows*(k_ + 1)/nt);
            }
            for (auto & x : th) { x.join(); }
        }

        // q in the chunk is [d_k, nc, n_head] (head-major), the FA output [d_v, n_head, nc] (query-major): the oracle above
        // reads q at (h*nc + i) and writes at (i*n_head + h), the same row order as the backend outputs
        const int64_t row0 = t0*n_head;
        if (sup) {
            accumulate(R.pairs["I-S"],  I.data(),  nullptr, S.data(),  d_v, n_rows, row0);
            accumulate(R.pairs["Is-S"], Is.data(), nullptr, S.data(),  d_v, n_rows, row0);
            accumulate(R.pairs["I-Is"], I.data(),  nullptr, Is.data(), d_v, n_rows, row0);
        }
        if (oracle) {
            accumulate(R.pairs["S-O"], S.data(), O.data(), nullptr, d_v, n_rows, row0);
            if (sup) {
                accumulate(R.pairs["I-O"],  I.data(),  O.data(), nullptr, d_v, n_rows, row0);
                accumulate(R.pairs["Is-O"], Is.data(), O.data(), nullptr, d_v, n_rows, row0);
            }
        }
    }
    ggml_backend_buffer_free(buf_kv);
    ggml_free(ctx_kv);

    if (!indexed_ok) {
        for (const char * n : { "I-S", "Is-S", "I-Is", "I-O", "Is-O" }) {
            R.pairs.erase(n);
        }
        R.info["indexed"] = "unsupported by the backend (I / Is not run)";
    }
    if (!oracle) {
        for (const char * n : { "S-O", "I-O", "Is-O" }) {
            R.pairs.erase(n);
        }
    }
    return R;
}

static inline json replay_json(const replay_result & R, const std::string & backend_name) {
    json j = R.info;
    j["backend"] = backend_name;
    j["membership"] = R.membership;
    json p = json::object();
    for (const auto & [n, s] : R.pairs) {
        p[n] = json { { "max_abs", s.max_abs }, { "nmse", s.den > 0 ? s.num/s.den : 0.0 }, { "rows_gt_1e-3", s.rows_gt },
                      { "argmax_row", s.argmax_row } };
    }
    j["pairs"] = p;
    if (!R.error.empty()) {
        j["error"] = R.error;
    }
    return j;
}

} // namespace qf
