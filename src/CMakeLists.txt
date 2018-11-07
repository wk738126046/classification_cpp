
ADD_EXECUTABLE(predict predict.cpp)
TARGET_LINK_LIBRARIES(predict mxnet ${OpenCV_LIBS} )

if(NOT MXNET_ROOT)
#    set(MXNET_ROOT $ENV{MXNET_ROOT})
    set(MXNET_ROOT "~/wk_ws/myTest/mxnet")
    message(STATUS "- mxnet path: ${MXNET_ROOT}")
endif()

# -- MXNet
include_directories(
        ${MXNET_ROOT}/include
        ${MXNET_ROOT}/cpp-package/include
        ${MXNET_ROOT}/3rdparty/dlpack/include
        ${MXNET_ROOT}/3rdparty/dmlc-core/include
        ${MXNET_ROOT}/3rdparty/mkldnn/include
        ${MXNET_ROOT}/3rdparty/mshadow
        ${MXNET_ROOT}/3rdparty/tvm/nnvm/include
        ${CMAKE_CURRENT_LIST_DIR}/src
)

ADD_EXECUTABLE(detect detect.cpp)
TARGET_LINK_LIBRARIES(detect mxnet ${OpenCV_LIBS} )