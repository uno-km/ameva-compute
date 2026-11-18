/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-AMEVA-Commercial
 */
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-AMEVA-Commercial
// ameva_repack_cache_check: the file format and the checks of the offline repack cache (ggml-ameva-repack-cache.h)
// without a GPU: write a cache, read it back, and show that every kind of wrong file is refused with its error code.
// Prints REPACK_CACHE_CHECK=PASS (n/n) or FAIL; exit status 0 only on PASS.
// Build (device, as ameva_repack_check): clang++ -O2 -std=c++17 -I <ggml>/include -I adapters/llama/ggml-ameva \
//   tests/ameva_repack_cache_check.cpp adapters/llama/ggml-ameva/ggml-ameva-repack.cpp <libggml-base.so> -o check
#include "ggml-ameva-repack-cache.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// the cache header declares it; the backend defines it
bool ameva_repack_enabled(void) { return true; }

static int g_total = 0, g_pass = 0;
static void check(bool ok, const char* what) {
    ++g_total;
    if (ok) ++g_pass;
    else printf("  FAIL: %s\n", what);
}
static bool has(const std::string& s, const char* code) { return s.find(code) != std::string::npos; }

static std::vector<uint8_t> slurp(const std::string& p) {
    std::vector<uint8_t> v;
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) return v;
    uint8_t b[65536];
    size_t n;
    while ((n = fread(b, 1, sizeof(b), f)) > 0) v.insert(v.end(), b, b + n);
    fclose(f);
    return v;
}
static void spit(const std::string& p, const std::vector<uint8_t>& v) {
    FILE* f = fopen(p.c_str(), "wb");
    fwrite(v.data(), 1, v.size(), f);
    fclose(f);
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : ".";
    const std::string good = dir + "/rpc_check_good.amvrp", bad = dir + "/rpc_check_bad.amvrp";
    using namespace ameva_rpc;

    // layout probe: stable and non-trivial
    check(layout_probe() != 0 && layout_probe() == layout_probe(), "layout probe is stable");

    // sample hash: sensitive to the sampled bytes, deterministic
    {
        std::vector<uint8_t> a(100000, 7), b = a;
        check(sample_hash(a.data(), a.size()) == sample_hash(b.data(), b.size()), "sample hash is deterministic");
        b[a.size() / 2] ^= 1;
        check(sample_hash(a.data(), a.size()) != sample_hash(b.data(), b.size()), "sample hash sees a middle byte");
        b = a;
        b[0] ^= 1;
        check(sample_hash(a.data(), a.size()) != sample_hash(b.data(), b.size()), "sample hash sees the first byte");
        b = a;
        b[a.size() - 1] ^= 1;
        check(sample_hash(a.data(), a.size()) != sample_hash(b.data(), b.size()), "sample hash sees the last byte");
    }

    // write a cache with three blobs
    const uint64_t key = 0x1122334455667788ull, msize = 123456789ull;
    std::vector<std::vector<uint8_t>> blobs;
    {
        Writer w;
        std::string e = w.open_tmp(good + ".tmp");
        check(e.empty(), "writer opens its temporary file");
        const size_t sizes[3] = { 5000, 64, 123457 };
        for (int i = 0; i < 3; ++i) {
            std::vector<uint8_t> b(sizes[i]);
            for (size_t j = 0; j < b.size(); ++j) b[j] = (uint8_t)(j * 31 + i * 7 + 1);
            Entry en;
            memset(&en, 0, sizeof(en));
            snprintf(en.name, sizeof(en.name), "blk.%d.weight", i);
            en.type = (uint32_t)GGML_TYPE_Q4_K;
            en.K = 256;
            en.rows = 10 + i;
            en.nbytes = b.size();
            en.src_hash = 1000 + i;
            en.off = w.reserve(b.size());
            // written in two parts, as the region writes of stage_rows are
            check(pwrite_all(w.fd(), en.off, b.data(), b.size() / 2) == 0 &&
                      pwrite_all(w.fd(), en.off + b.size() / 2, b.data() + b.size() / 2, b.size() - b.size() / 2) == 0,
                  "blob is written");
            w.add(en);
            blobs.push_back(b);
        }
        e = w.finish(good, key, msize);
        check(e.empty(), "writer finishes (table, header, fsync, rename)");
    }

    // read it back
    {
        Reader r;
        std::string e = r.open_file(good, key, msize);
        check(e.empty(), "reader accepts the file it was given");
        check(r.count() == 3, "reader sees three tensors");
        bool same = true;
        for (int i = 0; i < 3; ++i) {
            char name[32];
            snprintf(name, sizeof(name), "blk.%d.weight", i);
            const Entry* en = r.find(name);
            if (!en || en->nbytes != blobs[i].size() || en->src_hash != (uint64_t)(1000 + i) ||
                memcmp(r.blob(*en), blobs[i].data(), blobs[i].size()) != 0)
                same = false;
        }
        check(same, "every blob reads back byte for byte");
        check(r.find("blk.9.weight") == nullptr, "an absent tensor is a miss");
    }

    // every wrong file is refused with its code
    const std::vector<uint8_t> orig = slurp(good);
    check(orig.size() > kHeaderBytes, "good file is readable");
    auto refuse = [&](const std::vector<uint8_t>& bytes, uint64_t k, uint64_t sz, const char* code, const char* what) {
        spit(bad, bytes);
        Reader r;
        const std::string e = r.open_file(bad, k, sz);
        const bool ok = !e.empty() && has(e, code);
        if (!ok) printf("    got: %s\n", e.empty() ? "(accepted)" : e.c_str());
        check(ok, what);
    };
    {
        std::vector<uint8_t> v(orig.begin(), orig.end() - 7);
        refuse(v, key, msize, "AMU_ERR_REPACK_CACHE_CORRUPT", "truncated file");
        v = orig;
        v[0] = 'X';
        refuse(v, key, msize, "AMU_ERR_REPACK_CACHE_CORRUPT", "wrong magic");
        v = orig;
        v[offsetof(Header, layout_probe)] ^= 1; // header changed, header hash not
        refuse(v, key, msize, "AMU_ERR_REPACK_CACHE_CORRUPT", "header modified without its hash");
        Header h;
        memcpy(&h, orig.data(), sizeof(h));
        v = orig;
        v[h.table_off + 3] ^= 1;
        refuse(v, key, msize, "AMU_ERR_REPACK_CACHE_CORRUPT", "table modified");
        // a file of another layout: probe changed AND the header hash recomputed (a consistent file of an old build)
        Header o = h;
        o.layout_probe ^= 0xDEADBEEF;
        o.header_hash = fnv1a(&o, offsetof(Header, header_hash));
        v = orig;
        memcpy(v.data(), &o, sizeof(o));
        refuse(v, key, msize, "AMU_ERR_REPACK_CACHE_STALE", "file of another repacked layout (probe)");
        o = h;
        o.layout_version += 1;
        o.header_hash = fnv1a(&o, offsetof(Header, header_hash));
        v = orig;
        memcpy(v.data(), &o, sizeof(o));
        refuse(v, key, msize, "AMU_ERR_REPACK_CACHE_STALE", "file of another layout version");
        refuse(orig, key + 1, msize, "AMU_ERR_REPACK_CACHE_STALE", "file of another model (key)");
        refuse(orig, key, msize + 1, "AMU_ERR_REPACK_CACHE_STALE", "file of another model (size)");
    }

    remove(good.c_str());
    remove(bad.c_str());
    printf("REPACK_CACHE_CHECK=%s (%d/%d)\n", g_pass == g_total ? "PASS" : "FAIL", g_pass, g_total);
    return g_pass == g_total ? 0 : 1;
}
