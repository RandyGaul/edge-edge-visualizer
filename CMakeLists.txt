cmake_minimum_required(VERSION 3.10)
project(gauss_map_viz C)

add_executable(gauss_map_viz WIN32 gauss_map_viz.c)
target_link_libraries(gauss_map_viz opengl32 glu32 gdi32 user32)
