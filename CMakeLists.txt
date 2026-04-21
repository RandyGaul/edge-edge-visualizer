cmake_minimum_required(VERSION 3.10)
project(gauss_map_viz C)

if(EMSCRIPTEN)
    add_executable(gauss_map_viz gauss_map_viz.c)
    set_target_properties(gauss_map_viz PROPERTIES
        SUFFIX ".html"
        OUTPUT_NAME "index")
    target_link_options(gauss_map_viz PRIVATE
        "SHELL:-s USE_GLFW=3"
        "SHELL:-s LEGACY_GL_EMULATION=1"
        "SHELL:-s GL_UNSAFE_OPTS=0"
        "SHELL:-s ALLOW_MEMORY_GROWTH=1"
        "SHELL:-s GL_FFP_ONLY=1"
        -lGL)
else()
    add_executable(gauss_map_viz WIN32 gauss_map_viz.c)
    target_link_libraries(gauss_map_viz opengl32 glu32 gdi32 user32)
endif()
