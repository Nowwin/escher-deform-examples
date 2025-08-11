cmake_minimum_required(VERSION 3.12)
project(escher_deform_examples)
set(CMAKE_CXX_STANDARD 17)

# ---- Homebrew SDL2 (manual include + lib path) ----
set(SDL2_INCLUDE_DIRS
    /opt/homebrew/include
    /opt/homebrew/include/SDL2
)
set(SDL2_LIB_DIR /opt/homebrew/lib)
link_directories(${SDL2_LIB_DIR})

# ---- OpenCV via Homebrew ----
set(OpenCV_DIR "/opt/homebrew/opt/opencv/share/opencv4")
find_package(OpenCV REQUIRED)

# ---- Targets ----
add_executable(anteater anteater.cpp)  # no OpenMP
add_executable(lamp     lamp.cpp)      # with OpenMP

# ---- Common includes & libs applied to both ----
set(COMMON_INCLUDES ${SDL2_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
set(COMMON_LIBS SDL2 ${OpenCV_LIBS})

foreach(tgt anteater lamp)
  target_include_directories(${tgt} PRIVATE ${COMMON_INCLUDES})
  target_link_libraries(${tgt} PRIVATE ${COMMON_LIBS})
endforeach()

# ---- OpenMP for lamp (Clang + Homebrew libomp) ----
target_compile_options(lamp PRIVATE -Xpreprocessor -fopenmp)
target_include_directories(lamp PRIVATE /opt/homebrew/opt/libomp/include)
target_link_options(lamp PRIVATE -L/opt/homebrew/opt/libomp/lib -lomp)
# Optional: help the loader find libomp at runtime
# target_link_options(lamp PRIVATE "-Wl,-rpath,/opt/homebrew/opt/libomp/lib")
