// [TAG_QSA_FROZEN] llama-qsa-frozen — the frozen-M6 isolation tool (roadmap v5.1 stage 0a, math-code audit queue #3 = Codex #4).
// Spec: the header of ~/claude-tmp/codex/m6-frozen.sh. It separates the four effects mixed in the M6 gate's KL (membership,
// FA traversal order, kernel path, amplification) by freezing the inputs of one probe ubatch:
//
//   fill   -m GGUF --text FILE --depths D1,D2,... --probe 2048 --layers 3,23,47 --state-dir DIR --capture-dir DIR --out FILE
//          + ctx flags. Prefills the text in --probe-token ubatches (-ub must equal --probe); before the ubatch starting at
//          each depth d it saves the sequence state (DIR/state-<d>.bin, llama_state_seq_save_file), arms its own eval
//          callback for that probe ubatch [d, d + probe) and writes the flash-attention inputs of the chosen layers plus
//          the selection ids to <capture-dir>/cap-<d>-il<il>/ (layout: qsa-frozen-core.h). Disarmed, the callback asks
//          for no node: no extra splits, numerics as without the tool. --out: per-depth bytes / ms (JSON).
//   probe  -m GGUF --text FILE --state FILE --at D --probe 2048 --logits-last N --logits-out PREFIX + ctx flags.
//          Restores the state, decodes tokens [D, D + probe) as ONE ubatch and writes PREFIX.bin (f32 [N, n_vocab], the
//          last N positions) + PREFIX.json {n_last, n_vocab, first_pos, depth, variant, dtype, sha256, ...}. No callback is
//          installed, so LLAMA_QSA_DUMP_IDS=1 works. The variant comes from LLAMA_QSA_PREFILL / LLAMA_QSA_INDEXED_SORT.
//   replay --capture DIR --backend Vulkan0 --chunk 256 --oracle f64 --threads 16 --out FILE
//          ggml-only, per captured chunk: S = FA with the scan's top-k mask, I = indexed FA with the captured ids,
//          Is = the same with each row's ids sorted ascending, O = a host f64 oracle over the selected visible cells.
//          Membership is checked before any output; JSON = the m6-frozen-summary.py contract.
//          rc 0 = compared, 3 = membership failed (outputs not compared, JSON written), 1 = error.
//   fill / probe also take --ids FILE (whitespace-separated token ids) in place of --text, for a model without a
//   vocabulary (test-llama-archs' random qwen4exp in the smoke).
#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include "hash/hash.h"

#include "qsa-frozen-core.h"

// the attention cache's per-cell positions (internal API, as the tests do)
#include "llama-memory-hybrid.h"
#include "llama-kv-cache.h"

#include <cctype>
#include <chrono>
#include <clocale>
#include <cstdlib>
#include <string>
#include <vector>

using qf::json;

[[noreturn]] static void fail(const std::string & msg) {
    fprintf(stderr, "llama-qsa-frozen: %s\n", msg.c_str());
    exit(1);
}

static void usage() {
    fprintf(stderr,
        "usage:\n"
        "  llama-qsa-frozen fill   -m GGUF --text FILE|--ids FILE --depths D1,D2,.. --probe 2048 --layers 3,23,47 --state-dir DIR\n"
        "                          --capture-dir DIR --out FILE.json [ctx flags]\n"
        "  llama-qsa-frozen probe  -m GGUF --text FILE|--ids FILE --state FILE --at D --probe 2048 --logits-last N --logits-out PREFIX\n"
        "                          [ctx flags]\n"
        "  llama-qsa-frozen replay --capture DIR [--backend Vulkan0|CPU] [--chunk N] [--oracle f64|none] [--threads N]\n"
        "                          --out FILE.json\n");
    exit(2);
}

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

static std::string env_or(const char * k, const char * dflt) {
    const char * v = getenv(k);
    return v && *v ? v : dflt;
}

// --text FILE: tokenized with BOS as the server does; --ids FILE: whitespace-separated token ids (a model without a
// vocabulary, e.g. test-llama-archs' random qwen4exp, for the smoke)
static std::vector<llama_token> tokenize_file(llama_context * ctx, const std::string & path, bool ids) {
    const std::vector<uint8_t> raw = qf::read_file(path);
    const std::string s((const char *) raw.data(), raw.size());
    if (!ids) {
        return common_tokenize(ctx, s, /*add_special =*/ true, /*parse_special =*/ false);
    }
    std::vector<llama_token> out;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && isspace((unsigned char) s[i])) { ++i; }
        size_t j = i;
        while (j < s.size() && !isspace((unsigned char) s[j])) { ++j; }
        if (j > i) {
            out.push_back((llama_token) qf::parse_list(s.substr(i, j - i)).at(0));
        }
        i = j;
    }
    return out;
}

static std::vector<int32_t> attn_cell_positions(llama_context * ctx, int64_t n_kv) {
    auto * mem = dynamic_cast<llama_memory_hybrid *>(llama_get_memory(ctx));
    if (!mem) {
        fail("the context's memory is not a hybrid memory (qwen4exp expected)");
    }
    const llama_kv_cells & cells = mem->get_mem_attn()->get_cells(0);
    std::vector<int32_t> out((size_t) n_kv, -1);
    for (int64_t j = 0; j < n_kv && j < (int64_t) cells.size(); ++j) {
        out[j] = cells.is_empty((uint32_t) j) ? -1 : (int32_t) cells.pos_get((uint32_t) j);
    }
    return out;
}

static void decode_range(llama_context * ctx, llama_batch & batch, const std::vector<llama_token> & toks, int64_t p0, int64_t n,
                         int64_t n_last) {
    common_batch_clear(batch);
    for (int64_t i = 0; i < n; ++i) {
        common_batch_add(batch, toks[p0 + i], (llama_pos) (p0 + i), { 0 }, i >= n - n_last);
    }
    if (llama_decode(ctx, batch) != 0) {
        fail("llama_decode failed at position " + std::to_string(p0));
    }
}

// ---------------------------------------------------------------------------------------------------------------- fill

static int cmd_fill(common_params & params, const std::string & text, bool text_ids, const std::vector<int64_t> & depths_in,
                    int64_t probe, const std::vector<int64_t> & layers, const std::string & state_dir,
                    const std::string & cap_dir, const std::string & out) {
    std::vector<int64_t> depths = depths_in;
    std::sort(depths.begin(), depths.end());
    depths.erase(std::unique(depths.begin(), depths.end()), depths.end());
    for (int64_t d : depths) {
        if (d % probe != 0) {
            fail("depth " + std::to_string(d) + " is not a multiple of --probe " + std::to_string(probe));
        }
    }
    if (params.n_ubatch != (int32_t) probe || params.n_batch < (int32_t) probe) {
        fail("the probe ubatch must be ONE ubatch: -ub " + std::to_string(params.n_ubatch) + " must equal --probe and -b >= it");
    }
    if (system(("mkdir -p '" + state_dir + "' '" + cap_dir + "'").c_str()) != 0) {
        fail("cannot create the state / capture directories");
    }

    qf::capture cap;
    for (int64_t il : layers) {
        cap.layers.insert((int) il);
    }
    cap.dir         = cap_dir;
    cap.qsa_prefill = env_or("LLAMA_QSA_PREFILL", "scan");
    cap.git_rev     = llama_commit();

    params.cb_eval           = qf::capture_eval_cb;
    params.cb_eval_user_data = &cap;
    params.warmup            = false;

    llama_backend_init();
    llama_numa_init(params.numa);
    auto init = common_init_from_params(params);
    llama_context * ctx = init->context();
    if (!init->model() || !ctx) {
        fail("model / context init failed");
    }

    const std::vector<llama_token> toks = tokenize_file(ctx, text, text_ids);
    const int64_t end = depths.back() + probe;
    if ((int64_t) toks.size() < end) {
        fail("the text has " + std::to_string(toks.size()) + " tokens, the last probe needs " + std::to_string(end));
    }
    if ((int64_t) llama_n_ctx(ctx) < end) {
        fail("-c " + std::to_string(llama_n_ctx(ctx)) + " < the last depth + --probe = " + std::to_string(end));
    }
    LOG_INF("qsa-frozen fill: %zu text tokens, depths %s, probe %lld, layers %s, mode %s, rev %s\n", toks.size(),
            json(depths).dump().c_str(), (long long) probe, json(layers).dump().c_str(), cap.qsa_prefill.c_str(), cap.git_rev.c_str());

    llama_batch batch = llama_batch_init((int32_t) probe, 0, 1);
    json per_depth = json::array();
    const auto t_all = std::chrono::steady_clock::now();
    size_t di = 0;

    for (int64_t pos = 0; pos < end; pos += probe) {
        const bool is_probe = di < depths.size() && pos == depths[di];
        json rec;
        if (is_probe) {
            const std::string sf = state_dir + "/state-" + std::to_string(pos) + ".bin";
            const auto t0 = std::chrono::steady_clock::now();
            const size_t nb = llama_state_seq_save_file(ctx, sf.c_str(), 0, toks.data(), (size_t) pos);
            if (nb == 0) {
                fail("llama_state_seq_save_file failed for " + sf);
            }
            rec = json { { "depth", pos }, { "state_file", sf }, { "state_bytes", nb }, { "save_ms", ms_since(t0) } };
            cap.arm(pos);
        }
        const auto t1 = std::chrono::steady_clock::now();
        decode_range(ctx, batch, toks, pos, std::min(probe, end - pos), 0);
        if (is_probe) {
            rec["probe_decode_ms"] = ms_since(t1);
            int64_t n_kv = 0;
            for (const auto & [il, s] : cap.st) {
                n_kv = std::max(n_kv, s.n_kv);
            }
            std::vector<int32_t> tp((size_t) probe);
            for (int64_t i = 0; i < probe; ++i) {
                tp[i] = (int32_t) (pos + i);
            }
            rec["captures"] = cap.finish(attn_cell_positions(ctx, n_kv), tp);
            rec["n_kv"] = n_kv;
            LOG_INF("qsa-frozen fill: depth %lld captured (%s), state %.1f MiB\n", (long long) pos, rec["captures"].dump().c_str(),
                    rec["state_bytes"].get<double>()/1048576.0);
            per_depth.push_back(rec);
            di++;
        }
    }
    llama_batch_free(batch);

    const json doc = {
        { "tool", "llama-qsa-frozen" }, { "cmd", "fill" }, { "git_rev", cap.git_rev }, { "model", params.model.path },
        { "n_text_tokens", (int64_t) toks.size() }, { "probe", probe }, { "n_ctx", llama_n_ctx(ctx) },
        { "qsa_prefill", cap.qsa_prefill }, { "hadamard", cap.hadamard }, { "layers", layers }, { "depths", per_depth },
        { "fill_ms", ms_since(t_all) },
    };
    const std::string s = doc.dump(1) + "\n";
    qf::write_file(out, s.data(), s.size());
    LOG_INF("qsa-frozen fill: done in %.1f s -> %s\n", doc["fill_ms"].get<double>()/1000.0, out.c_str());
    return 0;
}

// --------------------------------------------------------------------------------------------------------------- probe

static int cmd_probe(common_params & params, const std::string & text, bool text_ids, const std::string & state, int64_t at, int64_t probe,
                     int64_t n_last, const std::string & prefix) {
    if (at < 0 || n_last <= 0 || n_last > probe) {
        fail("--at >= 0 and 0 < --logits-last <= --probe required");
    }
    if (params.n_ubatch < (int32_t) probe || params.n_batch < (int32_t) probe) {
        fail("the probe must decode as ONE ubatch: -b and -ub >= --probe");
    }
    params.warmup                = false;
    params.n_outputs_max_per_seq = 0;   // = n_outputs_max: every one of the last N positions has logits

    llama_backend_init();
    llama_numa_init(params.numa);
    auto init = common_init_from_params(params);
    llama_context * ctx = init->context();
    if (!init->model() || !ctx) {
        fail("model / context init failed");
    }
    const std::vector<llama_token> toks = tokenize_file(ctx, text, text_ids);
    if ((int64_t) toks.size() < at + probe) {
        fail("the text has " + std::to_string(toks.size()) + " tokens, the probe needs " + std::to_string(at + probe));
    }

    std::vector<llama_token> saved((size_t) std::max<int64_t>(at, 1));
    size_t n_saved = 0;
    if (llama_state_seq_load_file(ctx, state.c_str(), 0, saved.data(), saved.size(), &n_saved) == 0) {
        fail("llama_state_seq_load_file failed for " + state);
    }
    if ((int64_t) n_saved != at || !std::equal(saved.begin(), saved.begin() + at, toks.begin())) {
        fail("the state holds " + std::to_string(n_saved) + " tokens that are not the text's first --at " + std::to_string(at));
    }
    if (llama_memory_seq_pos_max(llama_get_memory(ctx), 0) != at - 1) {
        fail("restored sequence ends at " + std::to_string(llama_memory_seq_pos_max(llama_get_memory(ctx), 0)) +
             ", expected " + std::to_string(at - 1));
    }

    llama_batch batch = llama_batch_init((int32_t) probe, 0, 1);
    const auto t0 = std::chrono::steady_clock::now();
    decode_range(ctx, batch, toks, at, probe, n_last);
    const double dec_ms = ms_since(t0);

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(init->model()));
    std::vector<float> logits((size_t) n_last*n_vocab);
    for (int64_t i = 0; i < n_last; ++i) {
        const float * l = llama_get_logits_ith(ctx, (int32_t) (probe - n_last + i));
        if (!l) {
            fail("no logits at row " + std::to_string(probe - n_last + i));
        }
        std::copy(l, l + n_vocab, logits.begin() + i*n_vocab);
    }
    llama_batch_free(batch);

    const std::string prefill = env_or("LLAMA_QSA_PREFILL", "scan");
    const std::string sort    = env_or("LLAMA_QSA_INDEXED_SORT", "0");
    const std::string variant = prefill == "indexed" ? (sort == "1" ? "sorted" : "indexed") : prefill;

    qf::write_file(prefix + ".bin", logits.data(), logits.size()*sizeof(float));
    const json meta = {
        { "n_last", n_last }, { "n_vocab", n_vocab }, { "first_pos", at + probe - n_last }, { "depth", at }, { "probe", probe },
        { "variant", variant }, { "dtype", "f32" }, { "sha256", hash_sha256_hex(logits.data(), logits.size()*sizeof(float)) },
        { "decode_ms", dec_ms }, { "qsa_prefill", prefill }, { "indexed_sort", sort }, { "git_rev", llama_commit() },
        { "state", state },
    };
    const std::string s = meta.dump(1) + "\n";
    qf::write_file(prefix + ".json", s.data(), s.size());
    LOG_INF("qsa-frozen probe: depth %lld variant %s, %lld x %d logits -> %s.bin (%.1f ms)\n", (long long) at, variant.c_str(),
            (long long) n_last, n_vocab, prefix.c_str(), dec_ms);
    return 0;
}

// -------------------------------------------------------------------------------------------------------------- replay

static int cmd_replay(int argc, char ** argv) {
    std::string capdir, backend_name = "Vulkan0", oracle = "f64", out;
    int64_t chunk = 0;
    int threads = 16;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string { if (i + 1 >= argc) { fail("missing value for " + a); } return argv[++i]; };
        if      (a == "--capture") { capdir = val(); }
        else if (a == "--backend") { backend_name = val(); }
        else if (a == "--chunk")   { chunk = qf::parse_list(val()).at(0); }
        else if (a == "--oracle")  { oracle = val(); }
        else if (a == "--threads") { threads = (int) qf::parse_list(val()).at(0); }
        else if (a == "--out")     { out = val(); }
        else                       { fail("replay: unknown argument " + a); }
    }
    if (capdir.empty() || out.empty() || (oracle != "f64" && oracle != "none")) {
        usage();
    }
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(backend_name.c_str());
    if (!dev) {
        std::string have;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            have += std::string(" ") + ggml_backend_dev_name(ggml_backend_dev_get(i));
        }
        fail("no backend device " + backend_name + " (have:" + have + ")");
    }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    if (!be) {
        fail("cannot init " + backend_name);
    }
    if (auto * fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev),
                                                                                    "ggml_backend_set_n_threads")) {
        fn(be, threads);
    }

    const auto t0 = std::chrono::steady_clock::now();
    const qf::replay_result R = qf::replay(capdir, be, threads, oracle == "f64", chunk);
    json j = qf::replay_json(R, backend_name);
    j["replay_ms"] = ms_since(t0);
    j["oracle"] = oracle;
    j["capture"] = capdir;
    const std::string s = j.dump(1) + "\n";
    qf::write_file(out, s.data(), s.size());
    ggml_backend_free(be);

    fprintf(stderr, "qsa-frozen replay %s: membership %s%s\n", capdir.c_str(), R.membership.dump().c_str(),
            R.error.empty() ? "" : (" ERROR " + R.error).c_str());
    for (const auto & [n, p] : R.pairs) {
        fprintf(stderr, "  %-5s max|d| %.3e  NMSE %.3e  rows>1e-3 %lld\n", n.c_str(), p.max_abs, p.den > 0 ? p.num/p.den : 0.0,
                (long long) p.rows_gt);
    }
    if (!R.error.empty()) {
        const bool member_ok = !R.membership.is_null() && R.membership.value("unique", false) && R.membership.value("in_range", false);
        return (R.membership.is_null() || member_ok) ? 1 : 3;
    }
    return 0;
}

// ---------------------------------------------------------------------------------------------------------------- main

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    if (argc < 2) {
        usage();
    }
    const std::string cmd = argv[1];
    try {
        if (cmd == "replay") {
            return cmd_replay(argc, argv);
        }
        if (cmd != "fill" && cmd != "probe") {
            usage();
        }
        std::string text, state_dir, cap_dir, out, state, prefix;
        bool text_ids = false;
        std::vector<int64_t> depths, layers;
        int64_t probe = 2048, at = -1, n_last = 0;
        std::vector<char *> rest = { argv[0] };
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            auto val = [&]() -> std::string { if (i + 1 >= argc) { fail("missing value for " + a); } return argv[++i]; };
            if      (a == "--text")        { text = val(); }
            else if (a == "--ids")         { text = val(); text_ids = true; }
            else if (a == "--depths")      { depths = qf::parse_list(val()); }
            else if (a == "--probe")       { probe = qf::parse_list(val()).at(0); }
            else if (a == "--layers")      { layers = qf::parse_list(val()); }
            else if (a == "--state-dir")   { state_dir = val(); }
            else if (a == "--capture-dir") { cap_dir = val(); }
            else if (a == "--out")         { out = val(); }
            else if (a == "--state")       { state = val(); }
            else if (a == "--at")          { at = qf::parse_list(val()).at(0); }
            else if (a == "--logits-last") { n_last = qf::parse_list(val()).at(0); }
            else if (a == "--logits-out")  { prefix = val(); }
            else                           { rest.push_back(argv[i]); }
        }
        if (text.empty() || probe <= 0) {
            usage();
        }
        common_params params;
        if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_COMMON)) {
            return 1;
        }
        common_init();
        if (cmd == "fill") {
            if (depths.empty() || layers.empty() || state_dir.empty() || cap_dir.empty() || out.empty()) {
                usage();
            }
            return cmd_fill(params, text, text_ids, depths, probe, layers, state_dir, cap_dir, out);
        }
        if (state.empty() || prefix.empty()) {
            usage();
        }
        return cmd_probe(params, text, text_ids, state, at, probe, n_last, prefix);
    } catch (const std::exception & e) {
        fail(e.what());
    }
}
