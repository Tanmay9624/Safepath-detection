cmake_minimum_required(VERSION 3.15)
project(SafePathEdgePipeline CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------------------------------------------------------------
# Configuration for Windows vs Linux/Edge Devices
# -------------------------------------------------------------
if(WIN32)
    # Check for local deps folder first (self-contained build)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/deps/opencv/build")
        set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/opencv/build")
    elseif(EXISTS "C:/opencv/build")
        set(OpenCV_DIR "C:/opencv/build")
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/deps/onnxruntime")
        set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/onnxruntime")
    endif()
else()
    # Linux / Jetson standard installation path
    set(ONNXRUNTIME_DIR "/usr/local/include/onnxruntime")
endif()

# -------------------------------------------------------------
# Find OpenCV
# -------------------------------------------------------------
find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV: ${OpenCV_VERSION} at ${OpenCV_DIR}")

# -------------------------------------------------------------
# Find ONNX Runtime C++
# -------------------------------------------------------------
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h 
          PATHS ${ONNXRUNTIME_DIR} ${ONNXRUNTIME_DIR}/include)
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime 
             PATHS ${ONNXRUNTIME_DIR}/lib)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "Could NOT find ONNX Runtime headers or library in ${ONNXRUNTIME_DIR}. Please check the path.")
endif()

message(STATUS "Found ONNX Runtime Include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "Found ONNX Runtime Library: ${ONNXRUNTIME_LIBRARY}")

# -------------------------------------------------------------
# Build Executable
# -------------------------------------------------------------
add_executable(safepath pipeline.cpp)

target_include_directories(safepath PRIVATE ${OpenCV_INCLUDE_DIRS} ${ONNXRUNTIME_INCLUDE_DIR})

if(WIN32)
    # Windows MSVC handles threading natively (no -lpthread)
    target_link_libraries(safepath PRIVATE ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARY})
else()
    # POSIX threads on Linux/Jetson
    target_link_libraries(safepath PRIVATE ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARY} pthread)
endif()