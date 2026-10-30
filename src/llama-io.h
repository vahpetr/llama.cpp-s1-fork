#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct ggml_tensor;

class llama_io_write_i {
public:
    llama_io_write_i() = default;
    virtual ~llama_io_write_i() = default;

    virtual void write(const void * src, size_t size) = 0;
    virtual void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // [TAG_DISK_INCR] incremental capture (incremental() true): every KV cache range goes to write_tensor_kv with its
    //   first `clean` bytes (whole rows) belonging to cells not modified since mod_since() (a llama_kv_mod_clock value;
    //   0 = no earlier capture: clean is 0, the ranges still arrive there so the caller can record them). Otherwise the
    //   KV caches call write_tensor only. The stream bytes are the range's bytes either way
    virtual bool incremental() const { return false; }
    virtual uint64_t mod_since() const { return 0; }
    virtual void write_tensor_kv(ggml_tensor * tensor, size_t offset, size_t size, size_t /* clean */) {
        write_tensor(tensor, offset, size);
    }

    // bytes written so far
    virtual size_t n_bytes() = 0;

    void write_string(const std::string & str);
};

class llama_io_read_i {
public:
    llama_io_read_i() = default;
    virtual ~llama_io_read_i() = default;

    virtual void read(void * dst, size_t size) = 0;
    virtual void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // drop tensor data that has been read but not yet applied (e.g. when a restore fails)
    virtual void discard() {}

    // bytes read so far
    virtual size_t n_bytes() = 0;

    void read_string(std::string & str);

    // [TAG_STATE_READ_STATUS] why the read failed, a llama_state_read_status value (llama-ext.h) recorded by the memory
    //   reader at the failure site; the first reason wins, 0 = none recorded
    int status = 0;
    void set_status(int s) {
        if (status == 0) {
            status = s;
        }
    }
};
