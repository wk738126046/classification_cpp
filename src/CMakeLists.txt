
ADD_EXECUTABLE(predict predict.cpp)
TARGET_LINK_LIBRARIES(predict mxnet ${OpenCV_LIBS} )