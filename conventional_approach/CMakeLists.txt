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
    src/config.cc
)

# 视频批处理：531 帧 -> 标注视频 + CSV
add_executable(armor_video
    src/main_video.cc
    src/PnP_solver.cc
    include/func.cc
    src/config.cc
)


find_package( OpenCV REQUIRED )
include_directories( include/)
target_link_libraries( armor_conventional_detector PUBLIC ${OpenCV_LIBS} )
target_link_libraries( armor_video                PUBLIC ${OpenCV_LIBS} )

foreach(tgt armor_conventional_detector armor_video)
    # 配置文件的默认路径 = 仓库根目录/config/default.yaml，
    # 与当前工作目录无关
    target_compile_definitions(${tgt} PRIVATE
        PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/..")
endforeach()
