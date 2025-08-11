cmake_minimum_required(VERSION 3.12)
project(cpp_port)
set(CMAKE_CXX_STANDARD 17)

# SDL2 from Homebrew
include_directories(
    /opt/homebrew/include
    /opt/homebrew/include/SDL2
)
link_directories(/opt/homebrew/lib)

# Find and link OpenCV
set(OpenCV_DIR "/opt/homebrew/opt/opencv/share/opencv4")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cpp_port main.cpp)

# Link SDL2 and OpenCV
target_link_libraries(cpp_port SDL2 ${OpenCV_LIBS})
