#include <assert.h>

#include "CLI11.hpp"
#include "fmt/base.h"
#include "fmt/format.h"
#include "onnxruntime/core/session/experimental_onnxruntime_cxx_api.h"
#include "opencv2/core/mat.hpp"
#include "utils.hpp"
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <numeric>
#include <opencv2/core/hal/interface.h>
#include <opencv2/opencv.hpp>
#include <ostream>
#include <queue>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using namespace Ort::Experimental;

#define FAKE_INPUT_DATA 1

struct Options {
  std::string modelPath;
  std::string quantizedModelPath;
  std::string imagePath;
} Option;

void printModelInfo(Ort::Experimental::Session &session) {
  size_t num_output_nodes = session.GetOutputCount();
  std::vector<std::string> input_node_names = session.GetInputNames();
  std::vector<std::string> output_node_names = session.GetOutputNames();
  auto inputShapes = session.GetInputShapes();
  auto outputShapes = session.GetOutputShapes();

  // 迭代所有的输入节点
  for (int i = 0, num = session.GetInputCount(); i < num; i++) {
    Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    fmt::print("{} name: {} shape: {} type: {}\n", i, input_node_names[i],
               vectorToString(inputShapes[i]), static_cast<int>(type));
  }

  for (size_t i = 0; i < num_output_nodes; i++) {
    Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    fmt::print("{} name: {} shape: {} type: {}\n", i, output_node_names[i],
               vectorToString(outputShapes[i]), static_cast<int>(type));
  }
}

cv::Mat hwc2chw(const cv::Mat &image) {
  std::vector<cv::Mat> rgb_images;
  cv::split(image, rgb_images);

  // Stretch one-channel images to vector
  cv::Mat m_flat_r = rgb_images[0].reshape(1, 1);
  cv::Mat m_flat_g = rgb_images[1].reshape(1, 1);
  cv::Mat m_flat_b = rgb_images[2].reshape(1, 1);

  // Now we can rearrange channels if need
  cv::Mat matArray[] = {m_flat_r, m_flat_g, m_flat_b};

  cv::Mat flat_image;
  // Concatenate three vectors to one
  cv::hconcat(matArray, 3, flat_image);
  return flat_image;
}

#if FAKE_INPUT_DATA
std::tuple<std::vector<Ort::Value>, std::vector<Ort::Value>>
getInput(Ort::Experimental::Session &session, std::string imagePath) {
  auto inputShapes = session.GetInputShapes();
  auto outputShapes = session.GetOutputShapes();
  std::vector<Ort::Value> inputValue;
  for (auto &s : inputShapes) {
    int product = std::accumulate(s.begin(), s.end(), 1, std::multiplies<>());

    auto p = new float[product];
    std::fill(p, p + product, 10.0f);

    inputValue.push_back(Value::CreateTensor(p, product, s));
  }

  std::vector<Ort::Value> outputValue;
  for (auto &s : outputShapes) {
    outputValue.push_back(Value::CreateTensor<float>(s));
  }

  return {std::move(inputValue), std::move(outputValue)};
}
#else
std::tuple<std::vector<Ort::Value>, std::vector<Ort::Value>>
getInput(Ort::Experimental::Session &session, std::string imagePath) {
  auto inputShapes = session.GetInputShapes();
  auto outputShapes = session.GetOutputShapes();
  std::vector<Ort::Value> inputValue;
  assert(inputShapes.size() == 1 && "getInput");
  assert(inputShapes[0].size() == 4 && "getInput");
  for (auto &shape : inputShapes) {
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
      fmt::print("Could not open or find the image\n");
      return {};
    }
    cv::Mat imageResized;
    cv::resize(image, imageResized, cv::Size(shape[3], shape[2]));
    cv::Mat imageCHW;
    cv::cvtColor(imageResized, imageCHW, cv::COLOR_BGR2RGB);
    auto flat_image = hwc2chw(imageCHW);
    flat_image.convertTo(flat_image, CV_32FC3);

    int product =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());

    inputValue.push_back(
        Value::CreateTensor(flat_image.ptr<float>(), product, shape));
  }

  std::vector<Ort::Value> outputValue;
  for (auto &s : outputShapes) {
    outputValue.push_back(Value::CreateTensor<float>(s));
  }

  return {std::move(inputValue), std::move(outputValue)};
}
#endif

template <int K = 3>
auto runModel(Ort::Experimental::Session &session,
              std::vector<Ort::Value> &inputValue,
              std::vector<Ort::Value> &outputValue) {
  std::vector<std::string> input_node_names = session.GetInputNames();
  std::vector<std::string> output_node_names = session.GetOutputNames();

  session.Run(input_node_names, inputValue, output_node_names, outputValue);

  // 对结果进行softmax

  auto topKResult = topK(std::move(outputValue[0]), K);

  for (const auto &result : topKResult) {
    fmt::print("result: {}\n", topKResultToString(result));
  }
}

auto testRead(std::string imagePath) {
  cv::Mat image = cv::imread(imagePath);
  if (image.empty()) {
    fmt::print("Could not open or find the image\n");
    return;
  }
  cv::Mat imageResized;
  cv::resize(image, imageResized, cv::Size(224, 224));
  cv::Mat imageCHW;
  cv::cvtColor(imageResized, imageCHW, cv::COLOR_BGR2RGB);
  auto flat_image = hwc2chw(imageCHW);
}

int main(int argc, char *argv[]) {
  CLI::App app{"onnxruntime app"};

  app.add_option("-m,--model", Option.modelPath, "path to model")->required();
  app.add_option("-q,--quantized_model", Option.quantizedModelPath,
                 "path to quantized model");
  app.add_option("-i,--image", Option.imagePath, "path to image")->required();
  CLI11_PARSE(app, argc, argv);

  auto start = std::chrono::high_resolution_clock::now();
  {
    fmt::print("model path: {}\n", Option.modelPath);
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING);
    Ort::SessionOptions session_options;

    Ort::Experimental::Session session(env, Option.modelPath, session_options);

    printModelInfo(session);

    auto [inputValue, outputValue] = getInput(session, Option.imagePath);
    runModel(session, inputValue, outputValue);
  }

  if (!Option.quantizedModelPath.empty()) {
    fmt::print("quantized model path: {}\n", Option.quantizedModelPath);
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING);
    Ort::SessionOptions session_options;

    Ort::Experimental::Session session(env, Option.quantizedModelPath,
                                       session_options);

    printModelInfo(session);

    auto [inputValue, outputValue] = getInput(session, Option.imagePath);
    runModel(session, inputValue, outputValue);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  fmt::print("Elapsed time: {} seconds\n", elapsed.count());
  fmt::print("Done!\n");
  return 0;
}