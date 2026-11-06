#pragma once

#include "server-common.h"
#include "server-http.h"
#include "server-queue.h"
#include "server-mcp.h"

#include <atomic>
#include <functional>
#include <memory>

struct server_tool {
    std::string name;
    std::string display_name;
    bool permission_write = false;
    bool support_stream = false; // if true, output can be streamed
    bool uses_cwd = false;       // if true, the tool resolves paths and runs against the working directory

    virtual ~server_tool() = default;
    virtual json get_definition() const = 0;
    virtual std::string type() const { return "server"; }

    struct stream {
        server_response & qr;
        int id;
        std::function<bool()> alive;
        void push(const std::string & chunk);
    };
    virtual json invoke(json params, stream * st = nullptr) const = 0;

    json to_json() const;
};

// limits of the built-in tools, from --tools-read-max / --tools-exec-*
struct server_tools_limits {
    size_t read_max         = 256 * 1024;
    int    exec_timeout     = 60;
    int    exec_timeout_max = 600;
    size_t exec_output      = 64 * 1024;
    size_t exec_output_max  = 256 * 1024;
};

struct server_tools_runtime; // impl detail, defined in server-tools.cpp

struct server_tools {
    std::vector<std::unique_ptr<server_tool>> tools;

    // for streaming
    server_response queue_res;
    std::atomic<int> res_id{0};

    // set when --tools-runtime is configured; routes every tool call through an isolate
    std::unique_ptr<server_tools_runtime> runtime;

    void setup(const std::vector<std::string> & enabled_tools,
               server_mcp & mcp_mgr,
               const std::string & tools_runtime,
               const server_tools_limits & limits = {});

    server_http_context::handler_t handle_get;
    server_http_context::handler_t handle_post;

    server_tools();
    ~server_tools();
};
