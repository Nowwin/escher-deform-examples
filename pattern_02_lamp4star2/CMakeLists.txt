cmake_minimum_required(VERSION 3.12)
project(cpp_port)
set(CMAKE_CXX_STANDARD 17)

include_directories(
    /opt/homebrew/include
    /opt/homebrew/include/SDL2
)
link_directories(/opt/homebrew/lib)

set(OpenCV_DIR "/opt/homebrew/opt/opencv/share/opencv4")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/opt/homebrew/opt/libomp/lib -lomp")

add_executable(cpp_port main.cpp)

target_link_libraries(cpp_port SDL2 ${OpenCV_LIBS})
