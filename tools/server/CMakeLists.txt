include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# server-context containing the core server logic, used by llama-server and CLI

set(TARGET server-context)

add_library(${TARGET} STATIC
    server-chat.cpp
    server-chat.h
    server-task.cpp
    server-task.h
    server-queue.cpp
    server-queue.h
    server-common.cpp
    server-common.h
    server-context.cpp
    server-context.h
    server-disk-cache.cpp
    server-disk-cache.h
    server-stream.cpp
    server-stream.h
    server-tools.cpp
    server-tools.h
    server-mcp.cpp
    server-mcp.h
    server-schema.cpp
    server-schema.h
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# [TAG_DISK_PROMPT_CACHE] the disk prompt cache's fingerprint covers the server's state-handling C++ sources
# (checkpoint creation / restore, token serialization, the cache itself), not the webui: a webui-only rebuild
# keeps the stored conversations, any change here drops them
set(SERVER_STATE_SRC_HASH "")
foreach(SRC server-context.cpp server-common.cpp server-common.h server-task.cpp server-task.h server-disk-cache.cpp server-disk-cache.h)
    file(SHA256 ${CMAKE_CURRENT_SOURCE_DIR}/${SRC} H)
    string(APPEND SERVER_STATE_SRC_HASH ${H})
endforeach()
string(SHA256 SERVER_STATE_SRC_HASH "${SERVER_STATE_SRC_HASH}")
string(SUBSTRING "${SERVER_STATE_SRC_HASH}" 0 16 SERVER_STATE_SRC_HASH)
set_source_files_properties(server-disk-cache.cpp PROPERTIES COMPILE_DEFINITIONS "SERVER_STATE_SRC_HASH=\"${SERVER_STATE_SRC_HASH}\"")

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})

# llama-server-impl: server logic, reusable by app

set(TARGET llama-server-impl)

add_library(${TARGET}
    server.cpp
    server-http.cpp
    server-http.h
    server-models.cpp
    server-models.h
)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ../mtmd ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC server-context llama-ui cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-server executable

set(TARGET llama-server)

add_executable(${TARGET} main.cpp)
install(TARGETS ${TARGET} RUNTIME)

target_link_libraries(${TARGET} PRIVATE llama-server-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
