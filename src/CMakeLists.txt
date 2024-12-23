add_executable(onnx-app "main.cpp")
target_link_libraries(onnx-app PRIVATE onnxruntime fmt::fmt-header-only ${OpenCV_LIBS})

install(TARGETS onnx-app DESTINATION bin)