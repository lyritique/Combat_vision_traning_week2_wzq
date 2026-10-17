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

# 视频批处理：531 帧 -> 标注视频 + CSV
add_executable(armor_video
    src/main_video.cc
    src/PnP_solver.cc
    include/func.cc
)


find_package( OpenCV REQUIRED )
include_directories( include/)
target_link_libraries( armor_conventional_detector PUBLIC ${OpenCV_LIBS} )
target_link_libraries( armor_video                PUBLIC ${OpenCV_LIBS} )

# 把模型目录的绝对路径编进程序，避免"在 build/ 里运行就找不到 models/mlp.onnx"
target_compile_definitions( armor_conventional_detector PRIVATE
    ARMOR_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/models" )
target_compile_definitions( armor_video PRIVATE
    ARMOR_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/models" )
