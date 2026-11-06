#include "subproc.h"

bool common_subproc::is_supported() {
#ifdef LLAMA_SUBPROCESS
    return true;
#else
    return false;
#endif
}

#ifdef LLAMA_SUBPROCESS

#if !defined(_WIN32)
#include <cerrno>
#include <csignal>
#include <sys/wait.h>
#endif

static std::vector<char *> to_cstr_vec(const std::vector<std::string> & v) {
    std::vector<char *> r;
    r.reserve(v.size() + 1);
    for (const auto & s : v) {
        r.push_back(const_cast<char *>(s.c_str()));
    }
    r.push_back(nullptr);
    return r;
}

common_subproc::~common_subproc() {
    if (is_created) {
        subprocess_destroy(&proc);
        is_created = false;
    }
}

bool common_subproc::create(
        const std::vector<std::string> & args,
        int options,
        const std::vector<std::string> & env,
        const char * cwd) {
    auto argv = to_cstr_vec(args);

    int result;
    if (env.empty() && cwd == nullptr) {
        result = subprocess_create(argv.data(), options, &proc);
    } else {
        auto envp = to_cstr_vec(env);
        result = subprocess_create_ex(argv.data(), options, env.empty() ? nullptr : envp.data(), cwd, &proc);
    }

    is_created = result == 0;
    return is_created;
}

bool common_subproc::has_handle() const {
    if (!is_created) {
        return false;
    }
#if defined(_WIN32)
    return proc.hProcess != nullptr;
#else
    return proc.child > 0;
#endif
}

bool common_subproc::alive() {
    return is_created && subprocess_alive(&proc);
}

FILE * common_subproc::stdin_file() {
    return is_created ? subprocess_stdin(&proc) : nullptr;
}

FILE * common_subproc::stdout_file() {
    return is_created ? subprocess_stdout(&proc) : nullptr;
}

FILE * common_subproc::stderr_file() {
    return is_created ? subprocess_stderr(&proc) : nullptr;
}

void common_subproc::close_stdin() {
    if (is_created && proc.stdin_file) {
        fclose(proc.stdin_file);
        proc.stdin_file = nullptr;
    }
}

void common_subproc::terminate() {
    if (has_handle()) {
        subprocess_terminate(&proc);
    }
}

void common_subproc::kill_group(int sig) {
#if defined(_WIN32)
    (void) sig;
#else
    if (has_handle()) {
        kill(-proc.child, sig);
    }
#endif
}

bool common_subproc::exited() {
#if defined(_WIN32)
    return false;
#else
    if (!has_handle()) {
        return true;
    }
    siginfo_t info;
    info.si_pid = 0;
    int r;
    do {
        r = waitid(P_PID, (id_t) proc.child, &info, WEXITED | WNOHANG | WNOWAIT);
    } while (r == -1 && errno == EINTR);
    return r == 0 && info.si_pid != 0;
#endif
}

int common_subproc::join() {
    int exit_code = -1;
    if (is_created) {
        subprocess_join(&proc, &exit_code);
        subprocess_destroy(&proc);
        is_created = false;
    }
    return exit_code;
}

#else // !LLAMA_SUBPROCESS

common_subproc::~common_subproc() = default;

bool common_subproc::create(
        const std::vector<std::string> &,
        int,
        const std::vector<std::string> &,
        const char *) {
    (void)(proc);
    (void)(is_created);
    return false;
}

bool common_subproc::has_handle() const {
    return false;
}

bool common_subproc::alive() {
    return false;
}

FILE * common_subproc::stdin_file() {
    return nullptr;
}

FILE * common_subproc::stdout_file() {
    return nullptr;
}

FILE * common_subproc::stderr_file() {
    return nullptr;
}

void common_subproc::close_stdin() {
}

void common_subproc::terminate() {
}

void common_subproc::kill_group(int) {
}

bool common_subproc::exited() {
    return true;
}

int common_subproc::join() {
    return -1;
}

#endif // LLAMA_SUBPROCESS
