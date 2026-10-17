cmake_minimum_required(VERSION 3.11)
project(
    yolo_armor_detector
    VERSION 0.1.0
)

find_package(OpenCV REQUIRED)

# 两个方案共用的模块（PnP 解算 + 图片读入）直接引用兄弟目录里的实现，
# 避免复制一份出来各自维护。
set(SHARED_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../conventional_approach)

# 两个可执行程序共用的源文件
set(YOLO_COMMON_SOURCES
    include/preprocess.cc
    include/model_use.cc
    include/draw.cc
    ${SHARED_DIR}/src/PnP_solver.cc
)

add_executable( armor_yolo_detector
    src/yoloArmorDetector.cc
    ${YOLO_COMMON_SOURCES}
    ${SHARED_DIR}/include/input_image.cc
)

# 视频批处理：531 帧 -> 标注视频 + CSV
add_executable( armor_yolo_video
    src/main_video.cc
    ${YOLO_COMMON_SOURCES}
)

foreach(tgt armor_yolo_detector armor_yolo_video)
    target_include_directories( ${tgt} PRIVATE
        include
        ${SHARED_DIR}/include
    )
    target_link_libraries( ${tgt} PRIVATE ${OpenCV_LIBS} )

    # 把模型目录的绝对路径编进程序，避免"在 build/ 里运行就找不到模型"
    target_compile_definitions( ${tgt} PRIVATE
        YOLO_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/models" )
endforeach()
