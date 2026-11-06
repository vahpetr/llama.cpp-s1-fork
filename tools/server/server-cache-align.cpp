// [TAG_CACHE_ALIGN] byte-aligned prompt-cache splice - see server-cache-align.h
#include "server-cache-align.h"

#include "common.h"
#include "server-common.h"

#include <algorithm>
#include <cstring>

server_cache_align_vocab server_cache_align_vocab::from_llama(const llama_vocab * vocab) {
    server_cache_align_vocab v;
    const int32_t n = llama_vocab_n_tokens(vocab);
    v.piece.resize(n);
    v.structural.resize(n, 0);
    constexpr int attr_id_only = LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED | LLAMA_TOKEN_ATTR_UNKNOWN;
    for (int32_t t = 0; t < n; t++) {
        v.piece[t] = common_token_to_piece(vocab, t, /* special */ true);
        const int attr = (int) llama_vocab_get_attr(vocab, t);
        v.structural[t] = (attr & attr_id_only) != 0 || v.piece[t].empty();
    }
    return v;
}

server_cache_align_vocab server_cache_align_vocab::from_pieces(std::vector<std::string> pieces, const std::vector<llama_token> & structural_ids) {
    server_cache_align_vocab v;
    v.piece = std::move(pieces);
    v.structural.resize(v.piece.size(), 0);
    for (size_t t = 0; t < v.piece.size(); t++) {
        v.structural[t] = v.piece[t].empty();
    }
    for (const llama_token t : structural_ids) {
        if (t >= 0 && (size_t) t < v.piece.size()) {
            v.structural[t] = 1;
        }
    }
    return v;
}

bool server_cache_align_vocab::is_structural(llama_token t) const {
    return t < 0 || (size_t) t >= structural.size() || structural[t] != 0;
}

const char * server_cache_align_status_str(server_cache_align_status s) {
    switch (s) {
        case CACHE_ALIGN_NONE:            return "none";
        case CACHE_ALIGN_ALIGNED:         return "aligned";
        case CACHE_ALIGN_DECLINED_SPANS:  return "declined (message spans)";
        case CACHE_ALIGN_DECLINED_MEDIA:  return "declined (media)";
    }
    return "?";
}

namespace {

bool in_vocab(const server_cache_align_vocab & v, llama_token t) {
    return t >= 0 && (size_t) t < v.size();
}

// true if the ordinary pieces of `toks` from index k spell `lit` as a byte prefix (a literal special-token spelling)
bool spells_literal(const server_cache_align_vocab & v, const llama_tokens & toks, size_t k, const std::string & lit) {
    if (lit.empty()) {
        return false;
    }
    std::string acc;
    for (; k < toks.size() && acc.size() < lit.size(); k++) {
        if (!in_vocab(v, toks[k]) || v.is_structural(toks[k])) {
            return false;
        }
        acc += v.piece[toks[k]];
    }
    return acc.size() >= lit.size() && acc.compare(0, lit.size(), lit) == 0;
}

struct walk_sync {
    size_t n;     // index in N
    size_t c;     // index in C
    size_t bytes; // bytes past the LCP
    bool   usable;
};

} // namespace

server_cache_align_result server_cache_align(
        const llama_tokens & C,
        const llama_tokens & N,
        const server_cache_align_vocab & v,
        const server_cache_align_params & params) {
    server_cache_align_result r;

    // the id LCP
    size_t n = 0;
    if (params.n_lcp != SIZE_MAX) {
        n = std::min({ params.n_lcp, C.size(), N.size() });
    } else {
        const size_t m = std::min(C.size(), N.size());
        while (n < m && C[n] == N[n] && C[n] != LLAMA_TOKEN_NULL) {
            n++;
        }
    }
    r.n_lcp = r.n_c = r.n_n = n;

    const auto spans_of = [&](const llama_tokens & toks) {
        return params.delims != nullptr ? params.delims->split(toks) : common_chat_msg_spans{};
    };

    // the walk: advance the side that is behind by one piece; the buffers hold only the bytes since the last sync
    std::vector<walk_sync> syncs;
    syncs.push_back({ n, n, 0, true }); // the LCP itself: the no-op splice
    size_t i = n, j = n, base = 0;
    std::string bc, bn;

    const auto usable_at = [&](size_t ci, size_t nj) {
        if (nj == N.size()) {
            return true;                                                            // (c) the end of N
        }
        if (N[nj] == LLAMA_TOKEN_NULL || !in_vocab(v, N[nj]) || !v.is_structural(N[nj])) {
            return false;
        }
        return ci == C.size() || C[ci] == N[nj];                                    // (b) end of C / (a) common structural
    };

    const auto stop_on_structural = [&](llama_token a, llama_token b, size_t ci, size_t nj) {
        // a structural token on one side faces an ordinary spelling of the same bytes on the other
        const bool sa = v.is_structural(a), sb = v.is_structural(b);
        if (sa && !sb && in_vocab(v, a) && spells_literal(v, N, nj, v.piece[a])) {
            return CACHE_ALIGN_STOP_LITERAL;
        }
        if (sb && !sa && in_vocab(v, b) && spells_literal(v, C, ci, v.piece[b])) {
            return CACHE_ALIGN_STOP_LITERAL;
        }
        return CACHE_ALIGN_STOP_STRUCTURAL;
    };

    while (true) {
        if (bc.size() == bn.size()) {
            // a joint boundary (both buffers fully matched)
            base += bc.size();
            bc.clear();
            bn.clear();
            if (syncs.back().c != i || syncs.back().n != j) {
                syncs.push_back({ j, i, base, usable_at(i, j) });
            }
            if (i == C.size() || j == N.size()) {
                r.stop = CACHE_ALIGN_STOP_END;
                break;
            }
            const llama_token a = C[i], b = N[j];
            if (a == LLAMA_TOKEN_NULL || b == LLAMA_TOKEN_NULL) {
                r.stop = CACHE_ALIGN_STOP_MEDIA;
                break;
            }
            if (!in_vocab(v, a) || !in_vocab(v, b)) {
                r.stop = CACHE_ALIGN_STOP_STRUCTURAL;
                break;
            }
            if (a == b) {
                // same id: equal bytes, stays joint (also the only way past a structural token)
                bc = v.piece[a];
                bn = bc;
                i++;
                j++;
                continue;
            }
            if (v.is_structural(a) || v.is_structural(b)) {
                r.stop = stop_on_structural(a, b, i, j);
                break;
            }
            bc = v.piece[a];
            i++;
        } else if (bc.size() < bn.size()) {
            if (i == C.size()) {
                r.stop = CACHE_ALIGN_STOP_END;
                break;
            }
            const llama_token a = C[i];
            if (a == LLAMA_TOKEN_NULL) {
                r.stop = CACHE_ALIGN_STOP_MEDIA;
                break;
            }
            if (!in_vocab(v, a) || v.is_structural(a)) {
                // a structural token can only start at a joint boundary
                r.stop = in_vocab(v, a) && bn.compare(bc.size(), std::string::npos, v.piece[a], 0, bn.size() - bc.size()) == 0
                    ? CACHE_ALIGN_STOP_LITERAL : CACHE_ALIGN_STOP_STRUCTURAL;
                break;
            }
            bc += v.piece[a];
            i++;
        } else {
            if (j == N.size()) {
                r.stop = CACHE_ALIGN_STOP_END;
                break;
            }
            const llama_token b = N[j];
            if (b == LLAMA_TOKEN_NULL) {
                r.stop = CACHE_ALIGN_STOP_MEDIA;
                break;
            }
            if (!in_vocab(v, b) || v.is_structural(b)) {
                r.stop = in_vocab(v, b) && bc.compare(bn.size(), std::string::npos, v.piece[b], 0, bc.size() - bn.size()) == 0
                    ? CACHE_ALIGN_STOP_LITERAL : CACHE_ALIGN_STOP_STRUCTURAL;
                break;
            }
            bn += v.piece[b];
            j++;
        }
        const size_t k = std::min(bc.size(), bn.size());
        if (std::memcmp(bc.data(), bn.data(), k) != 0) {
            r.stop = CACHE_ALIGN_STOP_BYTES;
            break;
        }
    }

    // bytes of the common id prefix (rank by absolute matched bytes)
    size_t lcp_bytes = 0;
    if (params.prompt_bytes != nullptr && params.prompt_bytes->size() == N.size() + 1) {
        lcp_bytes = (*params.prompt_bytes)[n];
    } else {
        for (size_t k = 0; k < n; k++) {
            if (in_vocab(v, C[k])) {
                lcp_bytes += v.piece[C[k]].size();
            }
        }
    }

    const common_chat_msg_spans spans_n = spans_of(N);

    size_t last_null_n = SIZE_MAX;
    for (size_t k = N.size(); k-- > n;) {
        if (N[k] == LLAMA_TOKEN_NULL) {
            last_null_n = k;
            break;
        }
    }

    const auto accept = [&](size_t s) {
        r.n_n     = syncs[s].n;
        r.n_c     = syncs[s].c;
        r.n_bytes = lcp_bytes + syncs[s].bytes;
        r.syncs.clear();
        r.n_spans = 0;
        for (size_t k = 0; k <= s; k++) {
            r.syncs.emplace_back(syncs[k].n, syncs[k].c);
            if (k > 0) {
                const auto & p = syncs[k - 1];
                const auto & q = syncs[k];
                if (q.n - p.n != 1 || q.c - p.c != 1 || C[p.c] != N[p.n]) {
                    r.n_spans++;
                }
            }
        }
    };

    // the no-op answer
    const auto decline = [&](server_cache_align_status st) {
        accept(0);
        r.status = st;
        r.spans  = spans_n;
        return r;
    };

    int retries = 0;
    for (size_t s = syncs.size(); s-- > 1;) {
        if (!syncs[s].usable) {
            continue;
        }
        if (syncs[s].n == n && syncs[s].c == n) {
            break;
        }
        if (last_null_n != SIZE_MAX && last_null_n >= syncs[s].n) {
            return decline(CACHE_ALIGN_DECLINED_MEDIA);
        }
        accept(s);
        if (params.delims == nullptr) {
            r.status = CACHE_ALIGN_ALIGNED;
            return r;
        }
        // complete delimiter identity: spans of S == spans of N mapped through the splice, every start exact
        const llama_tokens S = server_cache_align_splice(C, N, r);
        const common_chat_msg_spans spans_s = spans_of(S);
        bool same = spans_s.spans.size() == spans_n.spans.size();
        for (size_t k = 0; same && k < spans_n.spans.size(); k++) {
            const auto mp = server_cache_align_map_pos(r, spans_n.spans[k].pos);
            same = !mp.snapped && spans_s.spans[k].pos == mp.pos && spans_s.spans[k].role == spans_n.spans[k].role;
        }
        if (same) {
            r.status = CACHE_ALIGN_ALIGNED;
            r.spans  = spans_s;
            return r;
        }
        r.n_tried++; // decline() keeps the count
        if (++retries > params.max_span_retries) {
            return decline(CACHE_ALIGN_DECLINED_SPANS);
        }
    }

    return decline(r.n_tried > 0 ? CACHE_ALIGN_DECLINED_SPANS : CACHE_ALIGN_NONE);
}

llama_tokens server_cache_align_splice(const llama_tokens & C, const llama_tokens & N, const server_cache_align_result & r) {
    if (r.n_c == r.n_n && r.n_c == r.n_lcp) {
        return N;
    }
    llama_tokens S;
    S.reserve(r.n_c + (N.size() - r.n_n));
    S.insert(S.end(), C.begin(), C.begin() + r.n_c);
    S.insert(S.end(), N.begin() + r.n_n, N.end());
    return S;
}

server_cache_align_pos server_cache_align_map_pos(const server_cache_align_result & r, size_t p) {
    if (p <= r.n_lcp) {
        return { p, false };
    }
    if (p >= r.n_n) {
        return { (size_t) ((int64_t) p + r.delta()), false };
    }
    // the last joint boundary at or before p (syncs ascend in both coordinates)
    auto it = std::upper_bound(r.syncs.begin(), r.syncs.end(), p,
            [](size_t x, const std::pair<size_t, size_t> & s) { return x < s.first; });
    --it; // syncs[0] = (n_lcp, n_lcp) < p
    return { it->second, it->first != p };
}

bool server_cache_align_prompt_from_text(const common_json & p) {
    if (p.is_string()) {
        return true;
    }
    if (p.is_object() && p.contains("prompt_string")) {
        return p.at("prompt_string").is_string();
    }
    return false; // numeric arrays and mixed arrays carry caller-chosen ids
}

bool server_cache_align_eligible(const server_cache_align_gate & g) {
    return g.enabled && g.from_text && g.cache_prompt && g.n_cmpl <= 1 && !g.is_child;
}

std::vector<size_t> server_cache_align_prefix_bytes(const llama_tokens & toks, const server_cache_align_vocab & v) {
    std::vector<size_t> out(toks.size() + 1, 0);
    for (size_t k = 0; k < toks.size(); k++) {
        out[k + 1] = out[k] + (in_vocab(v, toks[k]) ? v.piece[toks[k]].size() : 0);
    }
    return out;
}

std::vector<bool> server_cache_align_prompts_from_text(const common_json & prompt) {
    // mirrors tokenize_input_prompts: an array without numbers is a list of subprompts, anything else is one
    bool has_numbers = false;
    if (prompt.is_array()) {
        for (const auto & e : prompt) {
            has_numbers = has_numbers || e.is_number_integer();
        }
    }
    std::vector<bool> out;
    if (prompt.is_array() && !has_numbers) {
        for (const auto & p : prompt) {
            out.push_back(server_cache_align_prompt_from_text(p));
        }
    } else {
        out.push_back(server_cache_align_prompt_from_text(prompt));
    }
    return out;
}

server_cache_match server_cache_match_tokens(const server_tokens & C, const server_tokens & N, const server_cache_matcher * m) {
    server_cache_match out;
    out.n_lcp = C.get_common_prefix(N);
    out.n_c = out.n_n = out.n_bytes = out.n_lcp;
    if (m == nullptr || !m->active()) {
        return out; // v75
    }
    server_cache_align_params p;
    p.delims       = m->delims;
    p.n_lcp        = out.n_lcp;
    p.prompt_bytes = &m->prompt_bytes;
    const auto r = server_cache_align(C.get_tokens(), N.get_tokens(), *m->vocab, p);
    out.n_c     = r.n_c;
    out.n_n     = r.n_n;
    out.n_bytes = r.n_bytes;
    out.aligned = r.aligned();
    return out;
}
