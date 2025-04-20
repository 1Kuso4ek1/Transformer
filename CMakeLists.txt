cmake_minimum_required(VERSION 3.20)

project(LibTorchMNIST)

include(FetchContent)

set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcpu.zip")

FetchContent_Declare(
    libtorch
    URL ${LIBTORCH_URL}
    SOURCE_DIR libtorch
)
FetchContent_MakeAvailable(libtorch)

set(CMAKE_CXX_STANDARD 23)
find_package(Torch REQUIRED PATHS ${libtorch_SOURCE_DIR})

add_executable(nn main.cpp)

target_link_libraries(nn "${TORCH_LIBRARIES}")

include_directories(include/)
