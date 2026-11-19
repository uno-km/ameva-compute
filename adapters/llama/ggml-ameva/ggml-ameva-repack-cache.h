/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-AMEVA-Commercial
 */
#pragma once
// Offline repack cache: the weights of a model in the AMEVA repacked layout (ggml-ameva-repack.h), kept in a file so
// that a later load copies them into the GPU buffer instead of converting them again on the CPU (on the Galaxy A35 and
// A53 the CPU conversion ran at about 170 MB/s: Gemma-2-2B 13 s, docs/design/resident-runtime-and-load-path.md).
//
// Included by ggml-ameva-buffer.cpp only; nothing here is compiled unless that file includes it, and every function is
// a no-op while AMEVA_REPACK_CACHE is unset.
//
//   AMEVA_REPACK_CACHE=<dir>        enable; the cache file of a model is <dir>/<model key>.amvrp
//   AMEVA_REPACK_MODEL=<gguf path>  the model whose weights are loaded (the backend sees tensors, not files); required
//   AMEVA_REPACK_CACHE_REBUILD=1    record again although a file exists (the explicit way out of a stale-file error)
//
// File absent  -> the load converts as before and records the converted bytes; the file is completed when the first
//                 graph runs (written under a temporary name, fsync, rename), so a crashed load leaves no cache.
// File present -> every weight in it is copied from the file (no conversion, no read of the GGUF beyond a 12 KiB
//                 fingerprint per tensor). A file that does not match this build or this model is an error
//                 (AMU_ERR_REPACK_CACHE_*), never a silent conversion: the cache is a promise that the bytes are the
//                 ones this binary would have produced.
//
// Key and checks:
//   model key      FNV-1a over the GGUF size and three 1 MiB samples (start, middle, end)
//   layout probe   hash of the repacked bytes of a synthetic tensor of every repackable type, computed by the running
//                  code: any change of the layout changes it, whether or not kLayoutVersion was raised
//   per tensor     name, type, K, rows, size and a fingerprint of the GGUF bytes (start, middle, end of the tensor)
//   file           header and table hashes, total size (truncation)
// The bytes inside the blobs are not hashed on read (a bit flip in a blob is not detected; the checks above catch
// stale, foreign and truncated files). Layout of the file (little endian):
//   [0, 4096)        Header
//   blobs            one per tensor, 64-byte aligned, exactly the bytes of the tensor in the AMEVA buffer
//   table            n_tensors x Entry, at Header.table_off
#include "ggml.h"
#include "ggml-ameva-repack.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

bool ameva_repack_enabled(void);

namespace ameva_rpc {

static const uint32_t kFormat = 1;
// Raise when the bytes ameva_repack produces for any type change (the layout probe detects it regardless).
// 1: row-major Q/H/S regions, q4_K/q5_K/q6_K Q and H in 16-row tiles (2026-11-07)
static const uint32_t kLayoutVersion = 1;
static const uint64_t kHeaderBytes = 4096;

struct Header {
    char magic[8];
    uint32_t format;
    uint32_t layout_version;
    uint64_t layout_probe;
    uint64_t model_key;
    uint64_t model_size;
    uint64_t n_tensors;
    uint64_t table_off;
    uint64_t file_size;
    uint64_t table_hash;
    uint64_t header_hash; // over the bytes before this field
};
static_assert(sizeof(Header) == 80, "cache header layout");

struct Entry {
    char name[64];
    uint32_t type;
    uint32_t reserved;
    uint64_t K;
    uint64_t rows;
    uint64_t off;
    uint64_t nbytes;
    uint64_t src_hash;
};
static_assert(sizeof(Entry) == 112, "cache entry layout");

static const char kMagic[8] = { 'A', 'M', 'V', 'R', 'P', 'K', '0', '1' };

static inline uint64_t fnv1a(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* b = (const uint8_t*)p;
    for (size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= 1099511628211ull;
    }
    return h;
}

// Hash of the bytes of the repacked layout, made by the running code (see "layout probe" above).
static uint64_t layout_probe() {
    static const uint64_t value = [] {
        uint64_t h = 1469598103934665603ull;
        const int64_t K = 1024, rows = 20; // 20 rows: one full 16-row tile and a short one
        for (int t = 0; t < (int)GGML_TYPE_COUNT; ++t) {
            const enum ggml_type type = (enum ggml_type)t;
            AmevaRepackLayout L;
            if (!ameva_repack_layout(type, &L) || K % L.block_elems != 0) continue;
            const size_t nbytes = ggml_row_size(type, K) * (size_t)rows;
            std::vector<uint8_t> src(nbytes), dst(nbytes, 0);
            uint64_t x = 0x9E3779B97F4A7C15ull + (uint64_t)t;
            for (auto& b : src) {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                b = (uint8_t)x;
            }
            h = fnv1a(&t, sizeof(t), h);
            const bool ok = ameva_repack(type, K, rows, src.data(), dst.data());
            h = fnv1a(&ok, sizeof(ok), h);
            h = fnv1a(dst.data(), nbytes, h);
        }
        return h;
    }();
    return value;
}

// Fingerprint of n bytes at p: the size and up to three 4 KiB samples (start, middle, end).
static uint64_t sample_hash(const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    uint64_t h = fnv1a(&n, sizeof(n));
    const size_t S = 4096;
    if (n <= 3 * S) return fnv1a(b, n, h);
    h = fnv1a(b, S, h);
    h = fnv1a(b + n / 2 - S / 2, S, h);
    return fnv1a(b + n - S, S, h);
}

static inline uint64_t align64(uint64_t v) { return (v + 63) & ~(uint64_t)63; }

#if defined(__linux__)

// pwrite until everything is written; 0 or the errno.
static int pwrite_all(int fd, uint64_t off, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    while (n) {
        const ssize_t w = pwrite(fd, b, n, (off_t)off);
        if (w < 0) {
            if (errno == EINTR) continue;
            return errno;
        }
        b += w;
        off += (uint64_t)w;
        n -= (size_t)w;
    }
    return 0;
}

// Key of the model file: size and three 1 MiB samples. Empty error string on success.
static std::string model_key(const std::string& path, uint64_t* key, uint64_t* size) {
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return "AMU_ERR_REPACK_CACHE_CONFIG: cannot open AMEVA_REPACK_MODEL " + path + ": " + strerror(errno);
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        return "AMU_ERR_REPACK_CACHE_CONFIG: AMEVA_REPACK_MODEL " + path + " is empty or cannot be inspected";
    }
    const uint64_t n = (uint64_t)st.st_size, S = 1ull << 20;
    uint64_t h = fnv1a(&n, sizeof(n));
    std::vector<uint8_t> buf((size_t)std::min<uint64_t>(S, n));
    const uint64_t at[3] = { 0, n > S ? n / 2 - S / 2 : 0, n > S ? n - S : 0 };
    for (int i = 0; i < 3; ++i) {
        if (i > 0 && n <= S) break;
        const size_t want = (size_t)std::min<uint64_t>(S, n - at[i]);
        const ssize_t r = pread(fd, buf.data(), want, (off_t)at[i]);
        if (r != (ssize_t)want) {
            close(fd);
            return "AMU_ERR_REPACK_CACHE_IO: reading AMEVA_REPACK_MODEL " + path + " failed";
        }
        h = fnv1a(buf.data(), want, h);
    }
    close(fd);
    *key = h;
    *size = n;
    return "";
}

class Writer {
public:
    ~Writer() {
        if (fd_ >= 0) close(fd_);
    }
    std::string open_tmp(const std::string& tmp_path) {
        tmp_ = tmp_path;
        fd_ = open(tmp_.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0600);
        if (fd_ < 0) return "AMU_ERR_REPACK_CACHE_IO: cannot create " + tmp_ + ": " + strerror(errno);
        next_ = kHeaderBytes;
        return "";
    }
    int fd() const { return fd_; }
    uint64_t reserve(uint64_t nbytes) {
        const uint64_t off = next_;
        next_ = align64(next_ + nbytes);
        return off;
    }
    void add(const Entry& e) { entries_.push_back(e); }
    size_t count() const { return entries_.size(); }
    uint64_t bytes() const { return next_; }
    void abandon() {
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
        if (!tmp_.empty()) unlink(tmp_.c_str());
        tmp_.clear();
    }
    // Table, header, fsync, rename to final_path.
    std::string finish(const std::string& final_path, uint64_t key, uint64_t model_size) {
        const uint64_t table_off = align64(next_);
        const size_t table_bytes = entries_.size() * sizeof(Entry);
        Header h;
        memset(&h, 0, sizeof(h));
        memcpy(h.magic, kMagic, 8);
        h.format = kFormat;
        h.layout_version = kLayoutVersion;
        h.layout_probe = layout_probe();
        h.model_key = key;
        h.model_size = model_size;
        h.n_tensors = entries_.size();
        h.table_off = table_off;
        h.file_size = table_off + table_bytes;
        h.table_hash = fnv1a(entries_.data(), table_bytes);
        h.header_hash = fnv1a(&h, offsetof(Header, header_hash));
        std::vector<uint8_t> head(kHeaderBytes, 0);
        memcpy(head.data(), &h, sizeof(h));
        int e = pwrite_all(fd_, table_off, entries_.data(), table_bytes);
        if (!e) e = pwrite_all(fd_, 0, head.data(), head.size());
        if (!e && fsync(fd_) != 0) e = errno;
        if (e) return std::string("AMU_ERR_REPACK_CACHE_IO: writing the table of ") + tmp_ + ": " + strerror(e);
        close(fd_);
        fd_ = -1;
        if (rename(tmp_.c_str(), final_path.c_str()) != 0) {
            return std::string("AMU_ERR_REPACK_CACHE_IO: rename ") + tmp_ + " -> " + final_path + ": " + strerror(errno);
        }
        tmp_.clear();
        return "";
    }

private:
    int fd_ = -1;
    std::string tmp_;
    uint64_t next_ = kHeaderBytes;
    std::vector<Entry> entries_;
};

class Reader {
public:
    ~Reader() {
        if (map_) munmap((void*)map_, size_);
    }
    // Empty string when the file is a valid cache of this model for this build; otherwise the error.
    std::string open_file(const std::string& path, uint64_t key, uint64_t model_size) {
        const int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) return "AMU_ERR_REPACK_CACHE_IO: cannot open " + path + ": " + strerror(errno);
        struct stat st;
        if (fstat(fd, &st) != 0) {
            close(fd);
            return "AMU_ERR_REPACK_CACHE_IO: cannot inspect " + path;
        }
        size_ = (size_t)st.st_size;
        if (size_ < kHeaderBytes) {
            close(fd);
            return "AMU_ERR_REPACK_CACHE_CORRUPT: " + path + " is shorter than the header";
        }
        void* m = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (m == MAP_FAILED) return "AMU_ERR_REPACK_CACHE_IO: mmap of " + path + " failed: " + strerror(errno);
        map_ = (const uint8_t*)m;
        Header h;
        memcpy(&h, map_, sizeof(h));
        if (memcmp(h.magic, kMagic, 8) != 0) return "AMU_ERR_REPACK_CACHE_CORRUPT: " + path + " is not a repack cache (magic)";
        if (h.header_hash != fnv1a(&h, offsetof(Header, header_hash)))
            return "AMU_ERR_REPACK_CACHE_CORRUPT: header hash of " + path + " does not match";
        if (h.format != kFormat)
            return "AMU_ERR_REPACK_CACHE_STALE: " + path + " has file format " + std::to_string(h.format) + ", this build reads " +
                   std::to_string(kFormat);
        if (h.layout_version != kLayoutVersion || h.layout_probe != layout_probe()) {
            char b[160];
            snprintf(b, sizeof(b), "layout version %u probe %016llx, this build: version %u probe %016llx", h.layout_version,
                     (unsigned long long)h.layout_probe, kLayoutVersion, (unsigned long long)layout_probe());
            return "AMU_ERR_REPACK_CACHE_STALE: " + path + " was written for another repacked layout (" + b + ")";
        }
        if (h.model_key != key || h.model_size != model_size)
            return "AMU_ERR_REPACK_CACHE_STALE: " + path + " belongs to another model file (key or size differs)";
        if (h.file_size != size_)
            return "AMU_ERR_REPACK_CACHE_CORRUPT: " + path + " is " + std::to_string(size_) + " bytes, its header says " +
                   std::to_string(h.file_size) + " (truncated)";
        const uint64_t table_bytes = h.n_tensors * sizeof(Entry);
        if (h.table_off < kHeaderBytes || h.table_off > size_ || table_bytes > size_ - h.table_off)
            return "AMU_ERR_REPACK_CACHE_CORRUPT: table of " + path + " is outside the file";
        if (h.table_hash != fnv1a(map_ + h.table_off, (size_t)table_bytes))
            return "AMU_ERR_REPACK_CACHE_CORRUPT: table hash of " + path + " does not match";
        const Entry* t = (const Entry*)(map_ + h.table_off);
        index_.reserve((size_t)h.n_tensors);
        for (uint64_t i = 0; i < h.n_tensors; ++i) {
            if (t[i].off < kHeaderBytes || t[i].off > h.table_off || t[i].nbytes > h.table_off - t[i].off)
                return "AMU_ERR_REPACK_CACHE_CORRUPT: a blob of " + path + " is outside the data area";
            index_[std::string(t[i].name, strnlen(t[i].name, sizeof(t[i].name)))] = &t[i];
        }
        n_ = h.n_tensors;
        return "";
    }
    const Entry* find(const char* name) const {
        const auto it = index_.find(name);
        return it == index_.end() ? nullptr : it->second;
    }
    const uint8_t* blob(const Entry& e) const { return map_ + e.off; }
    uint64_t count() const { return n_; }

private:
    const uint8_t* map_ = nullptr;
    size_t size_ = 0;
    uint64_t n_ = 0;
    std::unordered_map<std::string, const Entry*> index_;
};

// -----------------------------------------------------------------------------
// Process state and the interface used by the buffer
// -----------------------------------------------------------------------------
enum class Mode { Off, Read, Write };

struct State {
    std::mutex mu;
    Mode mode = Mode::Off;
    std::string final_path, tmp_path;
    uint64_t key = 0, model_size = 0;
    Reader reader;
    Writer writer;
    std::unordered_set<std::string> recorded;
    uint64_t hits = 0, misses = 0, hit_bytes = 0, recorded_bytes = 0;
    std::vector<std::string> first_misses;
    std::atomic<bool> finished{ false };
};

static State& state() {
    static State s;
    return s;
}

static void cleanup_at_exit() {
    State& s = state();
    if (s.mode == Mode::Write && !s.finished.load()) s.writer.abandon();
}

static bool flag01(const char* name) {
    const char* v = getenv(name);
    if (!v || !*v || strcmp(v, "0") == 0) return false;
    if (strcmp(v, "1") == 0) return true;
    GGML_ABORT("AMU_ERR_REPACK_CACHE_CONFIG: %s must be 0 or 1 (got \"%s\")", name, v);
}

static bool init_once() {
    const char* dir = getenv("AMEVA_REPACK_CACHE");
    if (!dir || !*dir) return false;
    const char* model = getenv("AMEVA_REPACK_MODEL");
    if (!model || !*model)
        GGML_ABORT("AMU_ERR_REPACK_CACHE_CONFIG: AMEVA_REPACK_CACHE is set but AMEVA_REPACK_MODEL (the GGUF path) is not");
    if (!ameva_repack_enabled())
        GGML_ABORT("AMU_ERR_REPACK_CACHE_CONFIG: AMEVA_REPACK_CACHE needs repacking, but AMEVA_REPACK=0");
    struct stat st;
    if (stat(dir, &st) != 0 || !S_ISDIR(st.st_mode))
        GGML_ABORT("AMU_ERR_REPACK_CACHE_CONFIG: AMEVA_REPACK_CACHE=%s is not a directory", dir);
    const bool rebuild = flag01("AMEVA_REPACK_CACHE_REBUILD");
    State& s = state();
    std::string err = model_key(model, &s.key, &s.model_size);
    if (!err.empty()) GGML_ABORT("%s", err.c_str());
    char name[40];
    snprintf(name, sizeof(name), "%016llx.amvrp", (unsigned long long)s.key);
    s.final_path = std::string(dir) + "/" + name;
    struct stat fst;
    if (!rebuild && stat(s.final_path.c_str(), &fst) == 0) {
        err = s.reader.open_file(s.final_path, s.key, s.model_size);
        if (!err.empty())
            GGML_ABORT("%s\n(delete the file or set AMEVA_REPACK_CACHE_REBUILD=1 to record it again)", err.c_str());
        s.mode = Mode::Read;
        fprintf(stderr, "AMEVA_REPACK_CACHE mode=read file=%s tensors=%llu\n", s.final_path.c_str(),
                (unsigned long long)s.reader.count());
    } else {
        s.tmp_path = s.final_path + ".tmp." + std::to_string((long long)getpid());
        err = s.writer.open_tmp(s.tmp_path);
        if (!err.empty()) GGML_ABORT("%s", err.c_str());
        s.mode = Mode::Write;
        std::atexit(cleanup_at_exit);
        fprintf(stderr, "AMEVA_REPACK_CACHE mode=write file=%s (completed when the first graph runs)\n", s.final_path.c_str());
    }
    return true;
}

static bool active() {
    static const bool on = init_once();
    return on;
}

struct Ticket {
    enum Kind { None, Hit, Record } kind = None;
    const uint8_t* blob = nullptr; // Hit: the cached bytes of the tensor
    int fd = -1;                   // Record: the file and where the tensor's blob starts
    uint64_t blob_off = 0;
    Entry entry;
};

static thread_local Ticket* g_sink = nullptr;

// Decide how the whole-tensor write of the repacked tensor `t` (source bytes `src`) is served. May abort.
static void open_ticket(Ticket* tk, const struct ggml_tensor* t, const void* src, size_t offset, size_t size) {
    tk->kind = Ticket::None;
    if (!active()) return;
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    const size_t nbytes = ggml_nbytes(t);
    const bool whole = offset == 0 && size == nbytes;
    if (s.mode == Mode::Read) {
        const Entry* e = s.reader.find(t->name);
        if (!e) {
            ++s.misses;
            if (s.first_misses.size() < 3) s.first_misses.push_back(t->name);
            return;
        }
        if (!whole)
            GGML_ABORT("AMU_ERR_REPACK_CACHE_PARTIAL_WRITE: tensor %s is in the cache but was written in parts (offset %zu, size %zu of %zu)",
                       t->name, offset, size, nbytes);
        if (e->type != (uint32_t)t->type || e->K != (uint64_t)t->ne[0] || e->rows != (uint64_t)ggml_nrows(t) || e->nbytes != nbytes ||
            e->src_hash != sample_hash(src, size)) {
            GGML_ABORT("AMU_ERR_REPACK_CACHE_STALE: tensor %s differs from the cached one (type, shape, size or source fingerprint); %s was made from another model file",
                       t->name, s.final_path.c_str());
        }
        tk->kind = Ticket::Hit;
        tk->blob = s.reader.blob(*e);
        ++s.hits;
        s.hit_bytes += nbytes;
        return;
    }
    if (!whole)
        GGML_ABORT("AMU_ERR_REPACK_CACHE_PARTIAL_WRITE: tensor %s was written in parts (offset %zu, size %zu of %zu); it cannot be recorded",
                   t->name, offset, size, nbytes);
    if (!s.recorded.insert(t->name).second) return; // uploaded again: recorded once
    Entry& e = tk->entry;
    memset(&e, 0, sizeof(e));
    snprintf(e.name, sizeof(e.name), "%s", t->name);
    e.type = (uint32_t)t->type;
    e.K = (uint64_t)t->ne[0];
    e.rows = (uint64_t)ggml_nrows(t);
    e.nbytes = nbytes;
    e.src_hash = sample_hash(src, size);
    e.off = s.writer.reserve(nbytes);
    tk->kind = Ticket::Record;
    tk->fd = s.writer.fd();
    tk->blob_off = e.off;
}

static void close_ticket(Ticket* tk) {
    if (tk->kind != Ticket::Record) return;
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    s.writer.add(tk->entry);
    s.recorded_bytes += tk->entry.nbytes;
}

// Scope in which stage_rows (to the buffer) also records the bytes it writes.
struct SinkScope {
    explicit SinkScope(Ticket* tk) { g_sink = tk->kind == Ticket::Record ? tk : nullptr; }
    ~SinkScope() { g_sink = nullptr; }
};

// Bytes [rel, rel + n) of the recorded tensor's blob.
static void sink_put(uint64_t rel, const void* p, size_t n) {
    if (!g_sink) return;
    const int e = pwrite_all(g_sink->fd, g_sink->blob_off + rel, p, n);
    if (e) GGML_ABORT("AMU_ERR_REPACK_CACHE_IO: writing the cache failed: %s", strerror(e));
}

// Called when the first graph runs: the load is over.
static void finish() {
    if (!active()) return;
    State& s = state();
    if (s.finished.exchange(true)) return;
    std::lock_guard<std::mutex> lock(s.mu);
    if (s.mode == Mode::Read) {
        fprintf(stderr, "AMEVA_REPACK_CACHE read hits=%llu misses=%llu MiB=%.1f", (unsigned long long)s.hits,
                (unsigned long long)s.misses, (double)s.hit_bytes / (1024.0 * 1024.0));
        for (const auto& m : s.first_misses) fprintf(stderr, " miss:%s", m.c_str());
        fprintf(stderr, "\n");
        return;
    }
    if (s.writer.count() == 0) {
        s.writer.abandon();
        fprintf(stderr, "AMEVA_REPACK_CACHE write: no repacked tensor was recorded (no file written)\n");
        return;
    }
    const size_t n = s.writer.count();
    const std::string err = s.writer.finish(s.final_path, s.key, s.model_size);
    if (!err.empty()) GGML_ABORT("%s", err.c_str());
    fprintf(stderr, "AMEVA_REPACK_CACHE write: completed %s tensors=%zu MiB=%.1f\n", s.final_path.c_str(), n,
            (double)s.recorded_bytes / (1024.0 * 1024.0));
}

#else // !__linux__: the cache needs POSIX file calls

struct Ticket {
    enum Kind { None, Hit, Record } kind = None;
    const uint8_t* blob = nullptr;
};
static bool active() {
    const char* d = getenv("AMEVA_REPACK_CACHE");
    if (d && *d) GGML_ABORT("AMU_ERR_REPACK_CACHE_CONFIG: AMEVA_REPACK_CACHE is supported on Linux and Android only");
    return false;
}
static void open_ticket(Ticket* tk, const struct ggml_tensor*, const void*, size_t, size_t) {
    tk->kind = Ticket::None;
    (void)active();
}
static void close_ticket(Ticket*) {}
struct SinkScope {
    explicit SinkScope(Ticket*) {}
};
static void sink_put(uint64_t, const void*, size_t) {}
static void finish() { (void)active(); }

#endif

} // namespace ameva_rpc
