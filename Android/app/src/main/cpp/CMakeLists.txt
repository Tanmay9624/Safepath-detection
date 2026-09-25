# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html.
# For more examples on how to use CMake, see https://github.com/android/ndk-samples.

cmake_minimum_required(VERSION 3.22.1)

project("myapplication")

# Include ONNX Runtime headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libs/onnxruntime/headers)

# Configure OpenCV
set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sdk/native/jni)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/onnxruntime/jni/arm64-v8a/libonnxruntime.so)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
add_library(${CMAKE_PROJECT_NAME} SHARED
        native-lib.cpp)

# Specifies libraries CMake should link to your target library.
target_link_libraries(${CMAKE_PROJECT_NAME}
        android
        log
        onnxruntime
        ${OpenCV_LIBS})