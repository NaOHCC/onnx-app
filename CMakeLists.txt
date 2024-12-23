cmake_minimum_required(VERSION 3.13)

project(onnxruntime_samples C CXX)
string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")

# option(LIBPNG_ROOTDIR "libpng root dir")
# option(ONNXRUNTIME_ROOTDIR "onnxruntime root dir")

set(CMAKE_CXX_STANDARD 17)


include(FetchContent)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://gh.llkk.cc/https://github.com/fmtlib/fmt
  GIT_TAG        11.0.2) # 10.2.1
FetchContent_MakeAvailable(fmt)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(include)
add_subdirectory(src)