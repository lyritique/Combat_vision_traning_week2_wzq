cmake_minimum_required(VERSION 3.11)
project(
    armor_conventional_detector
    VERSION 0.1.0
)
add_executable(armor_conventional_detector
    src/main.cc
    src/PnP_solver.cc
    include/func.cc
    include/input_image.cc
)


find_package( OpenCV REQUIRED )
include_directories( include/)
target_link_libraries( armor_conventional_detector PUBLIC ${OpenCV_LIBS} )
