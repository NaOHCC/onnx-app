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
#include <filesystem>
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

#include "indicators.hpp"

using namespace Ort::Experimental;

// #define FAKE_INPUT_DATA 1

struct Options {
  std::string modelPath;
  std::string quantizedModelPath;
  std::string imagePath;
  std::string datasetDir;
  bool verbose = false;
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
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
      fmt::print("Could not open or find the image\n");
      return {};
    }
    cv::Mat imageResized;
    cv::resize(image, imageResized, cv::Size(shape[3], shape[2]));
    cv::Mat imageCHW;
    cv::cvtColor(imageResized, imageCHW, cv::COLOR_BGR2RGB);
    auto flat_image = hwc2chw(imageCHW);
    flat_image.convertTo(flat_image, CV_32F, 1.0 / 255.0);
    int product =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    // 打印flat_image前10个元素
    // for (int i = 0; i < 10; i++) {
    //   fmt::print("flat_image: {}\n", flat_image.at<float>(i));
    // }

    // fmt::print("type: {}\n", flat_image.type());

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

  if (Option.verbose) {
    for (const auto &result : topKResult) {
      fmt::print("result: {}\n", topKResultToString(result));
    }
  }
  auto top1 = topKResult[0][0];
  return top1;
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

ImageInfo parseString(const std::string &str, const std::string &splitters) {
  std::vector<std::string> result;
  std::string current = "";
  for (size_t i = 0; i < str.size(); i++) {
    if (splitters.find(str[i]) != std::string::npos) {
      if (!current.empty()) {
        result.push_back(current);
        current = "";
      }
      continue;
    }
    current += str[i];
  }
  if (!current.empty())
    result.push_back(current);
  return ImageInfo{std::stoi(result[1]), std::stoi(result[3])};
}

class DatasetRunner {
  std::string datasetDir;
  const std::vector<std::string> &files;
  const std::map<std::string, ImageInfo> &m;

  Ort::Experimental::Session &session;

public:
  DatasetRunner(std::string datasetDir, const std::vector<std::string> &files,
                const std::map<std::string, ImageInfo> &m,
                Ort::Experimental::Session &session)
      : datasetDir(datasetDir), files(files), m(m), session(session) {}

  void printModelInfo() { ::printModelInfo(session); }

  float run() {

    using namespace indicators;
    BlockProgressBar bar{
        option::BarWidth{80},
        option::Start{"["},
        option::End{"]"},
        option::ForegroundColor{Color::white},
        option::ShowElapsedTime{true},
        option::ShowRemainingTime{true},
        option::FontStyles{std::vector<FontStyle>{FontStyle::bold}},
        option::MaxProgress{files.size()}};

    unsigned correct = 0;
    int i = 0;
    for (const std::string &f : files) {
      auto info = m.at(f);
      auto [inputValue, outputValue] = getInput(session, f);
      auto top1 = runModel(session, inputValue, outputValue);
      if (info.label == top1.label) {
        correct += 1;
      }
      bar.tick();
      i++;
      // auto postfixText = fmt::format("acc: {}", (float)correct / i);
      // bar.set_option(option::PostfixText{postfixText});
      //

      // bar.set_option(option::PostfixText{std::string("acc: ") +
      //                                    std::to_string((float)correct /
      //                                    i)});
    }
    bar.mark_as_completed();
    indicators::show_console_cursor(true);
    return (float)correct / files.size();
  }
};

int main(int argc, char *argv[]) {
  CLI::App app{"onnxruntime app"};

  app.add_option("-m,--model", Option.modelPath, "path to model")->required();
  app.add_option("-q,--quantized_model", Option.quantizedModelPath,
                 "path to quantized model");
  auto group = app.add_option_group("data");
  group->add_option("-i,--image", Option.imagePath, "path to image");
  group->add_option("-d,--dataaset-dir", Option.datasetDir,
                    "path to dataset dir");
  app.add_flag("-v,--verbose", Option.verbose, "verbose mode");

  group->require_option(1);

  CLI11_PARSE(app, argc, argv);
  auto start = std::chrono::high_resolution_clock::now();
  if (!Option.imagePath.empty()) {
    {
      fmt::print("model path: {}\n", Option.modelPath);
      Ort::Env env(ORT_LOGGING_LEVEL_WARNING);
      Ort::SessionOptions session_options;

      Ort::Experimental::Session session(env, Option.modelPath,
                                         session_options);

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
  }

  if (!Option.datasetDir.empty()) {
    std::vector<std::string> files;
    for (const auto &entry :
         std::filesystem::directory_iterator(Option.datasetDir)) {
      files.push_back(entry.path().string());
    }

    // parse file name
    std::map<std::string, ImageInfo> m;
    for (const std::string &f : files) {
      auto info = parseString(f, "_.");
      m[f] = info;
    }

    {
      fmt::print("model path: {}\n", Option.modelPath);
      Ort::Env env(ORT_LOGGING_LEVEL_WARNING);
      Ort::SessionOptions session_options;

      Ort::Experimental::Session session(env, Option.modelPath,
                                         session_options);

      DatasetRunner runner(Option.datasetDir, files, m, session);
      runner.printModelInfo();
      float accuracy = runner.run();
      fmt::print("accuracy: {}\n", accuracy);
    }

    if (!Option.quantizedModelPath.empty()) {
      fmt::print("quantized model path: {}\n", Option.quantizedModelPath);
      Ort::Env env(ORT_LOGGING_LEVEL_WARNING);
      Ort::SessionOptions session_options;

      Ort::Experimental::Session session(env, Option.quantizedModelPath,
                                         session_options);

      DatasetRunner runner(Option.datasetDir, files, m, session);
      runner.printModelInfo();
      float accuracy = runner.run();
      fmt::print("accuracy: {}\n", accuracy);
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  fmt::print("Elapsed time: {} seconds\n", elapsed.count());
  fmt::print("Done!\n");
  return 0;
}