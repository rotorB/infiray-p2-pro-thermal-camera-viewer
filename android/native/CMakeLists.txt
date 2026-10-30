cmake_minimum_required(VERSION 3.22.1)
project(p2prouvc C)

add_library(p2prouvc SHARED raw_uvc.c)
target_compile_options(p2prouvc PRIVATE -Wall -Wextra -Werror)
# libuvc.so comes from UVCAndroid and is resolved at runtime with dlopen.
target_link_libraries(p2prouvc log dl)
