cmake_minimum_required(VERSION 3.14)
project(gauss_map_viz C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(EMSCRIPTEN)
    # Web build: Emscripten ships SDL2 as a port (-sUSE_SDL=2) and WebGL2
    # (= GLES 3.0) via -sFULL_ES3=1. We deliver a single .html/.js/.wasm
    # shaped by web/shell.html.
    add_executable(gauss_map_viz gauss_map_viz.c)
    set_target_properties(gauss_map_viz PROPERTIES
        SUFFIX ".html"
        OUTPUT_NAME "index")
    target_compile_options(gauss_map_viz PRIVATE
        "SHELL:-s USE_SDL=2")
    target_link_options(gauss_map_viz PRIVATE
        "SHELL:-s USE_SDL=2"
        "SHELL:-s FULL_ES3=1"
        "SHELL:-s MIN_WEBGL_VERSION=2"
        "SHELL:-s MAX_WEBGL_VERSION=2"
        "SHELL:-s ALLOW_MEMORY_GROWTH=1"
        "SHELL:-s INITIAL_MEMORY=32MB"
        "SHELL:-s STACK_SIZE=4MB"
        "SHELL:-s EXIT_RUNTIME=0"
        --shell-file=${CMAKE_SOURCE_DIR}/web/shell.html)
else()
    # Desktop: fetch SDL2 from source (shallow clone of a pinned tag).
    include(FetchContent)
    set(SDL_STATIC ON  CACHE BOOL "" FORCE)
    set(SDL_SHARED OFF CACHE BOOL "" FORCE)
    set(SDL_TEST   OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        SDL2
        GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
        GIT_TAG        release-2.30.11
        GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(SDL2)

    add_executable(gauss_map_viz gauss_map_viz.c)
    target_link_libraries(gauss_map_viz PRIVATE SDL2::SDL2-static SDL2::SDL2main)
    if(WIN32)
        target_link_libraries(gauss_map_viz PRIVATE opengl32)
        # Launch as a GUI app on Windows (no console window).
        set_target_properties(gauss_map_viz PROPERTIES WIN32_EXECUTABLE TRUE)
    else()
        find_package(OpenGL REQUIRED)
        target_link_libraries(gauss_map_viz PRIVATE OpenGL::GL)
    endif()
endif()
