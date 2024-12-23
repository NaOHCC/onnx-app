add_executable(onnx-app "main.cpp")
target_link_libraries(onnx-app PRIVATE onnxruntime fmt::fmt ${OpenCV_LIBS})

install(TARGETS onnx-app DESTINATION bin)