cmake_minimum_required(VERSION 3.11)
project(
    armor_conventional_detector
    VERSION 0.1.0
)
add_executable(armor_conventional_detector
    src/main.cc
    include/func.cc
    include/input_image.cc
)

# 视频批处理：跑 531 帧，输出标注视频 + CSV
add_executable(armor_video
    src/main_video.cc
    include/func.cc
)


find_package( OpenCV REQUIRED )
include_directories( include/)
target_link_libraries( armor_conventional_detector PUBLIC ${OpenCV_LIBS} )
target_link_libraries( armor_video                PUBLIC ${OpenCV_LIBS} )
