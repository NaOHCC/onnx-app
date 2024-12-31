#pragma once

#include "fmt/format.h"
#include "onnxruntime/core/session/experimental_onnxruntime_cxx_api.h"
#include <algorithm>
#include <assert.h>
#include <queue>
#include <sstream>
#include <utility>
#include <vector>

using Index = int;
using Label = int;

template <class T> struct TopKElement {
  T probability;
  Label label;
};

struct ImageInfo {
  Index index;
  Label label;
};

template <typename T>
std::vector<TopKElement<T>> topK(const std::vector<T> &data, int k) {
  if (k <= 0 || data.empty()) {
    return {};
  }

  using Element = TopKElement<T>; // {value, index}
  auto compare = [](const Element &a, const Element &b) {
    return a.probability < b.probability; // 最大堆比较：按值降序排列
  };
  std::priority_queue<Element, std::vector<Element>, decltype(compare)> maxHeap(
      compare);

  for (int i = 0; i < data.size(); ++i) {
    maxHeap.push({data[i], i});
  }

  std::vector<Element> result;
  for (int i = 0; i < k && !maxHeap.empty(); ++i) {
    result.push_back(maxHeap.top());
    maxHeap.pop();
  }

  return result;
}

template <typename T> auto softmax(std::vector<T> &&input) {
  T rowmax = *std::max_element(input.begin(), input.end());
  T sum = 0.0f;
  for (size_t i = 0; i != input.size(); ++i) {
    sum += input[i] = std::exp(input[i] - rowmax);
  }
  for (size_t i = 0; i != input.size(); ++i) {
    input[i] = input[i] / sum;
  }
  return input;
}

inline auto topK(Ort::Value tensor, int k) {
  auto typeAndShape = tensor.GetTensorTypeAndShapeInfo();

  auto shape = typeAndShape.GetShape();

  assert(shape.size() == 2 && "ArgMax2D");

  auto rows = shape[0];
  auto cols = shape[1];
  auto tensor_data = tensor.GetTensorData<float>();
  std::vector<std::vector<TopKElement<float>>> result(rows);
  for (int i = 0; i < rows; ++i) {
    auto softmaxed = softmax(std::vector<float>(tensor_data + i * cols,
                                                tensor_data + (i + 1) * cols));
    result[i] = topK(softmaxed, k);
  }

  return result;
}

inline int ArgMaxRow(const std::vector<float> &row) {
  return std::distance(row.begin(), std::max_element(row.begin(), row.end()));
}

// ArgMax for a 2D Tensor
inline std::vector<int> ArgMax2D(Ort::Value tensor) {
  auto typeAndShape = tensor.GetTensorTypeAndShapeInfo();

  auto shape = typeAndShape.GetShape();

  assert(shape.size() == 2 && "ArgMax2D");

  auto rows = shape[0];
  auto cols = shape[1];
  auto tensor_data = tensor.GetTensorData<float>();
  std::vector<int> result(rows);

  for (int i = 0; i < rows; ++i) {
    auto row_begin = tensor_data + i * cols;
    auto row_end = row_begin + cols;
    result[i] = ArgMaxRow(std::vector<float>(row_begin, row_end));
  }

  return result;
}

template <class T> auto vectorToString(const std::vector<T> &v) -> std::string {
  std::ostringstream oss;
  oss << "(";
  for (size_t i = 0; i < v.size(); ++i) {
    oss << v[i];
    if (i != v.size() - 1) {
      oss << ", ";
    }
  }
  oss << ")";
  return oss.str();
};

template <class T>
auto topKResultToString(const std::vector<TopKElement<T>> &topKResult) {
  std::string result = "[";
  for (size_t i = 0; i < topKResult.size(); ++i) {
    result += fmt::format("(label {}, probability: {})", topKResult[i].label,
                          topKResult[i].probability);
    if (i != topKResult.size() - 1) {
      result += ", ";
    }
  }
  result += "]";
  return result;
}

// tensor is 1D
template <class ElT> inline void printTensor(Ort::Value &tensor) {
  auto typeAndShape = tensor.GetTensorTypeAndShapeInfo();
  auto shape = typeAndShape.GetShape();
  auto tensor_data = tensor.GetTensorData<ElT>();
  fmt::print("shape: {}\n", vectorToString(shape));
  fmt::print("data: [");
  for (int i = 0; i < shape[0]; ++i) {
    fmt::print("{}", tensor_data[i]);
    if (i != shape[0] - 1) {
      fmt::print(", ");
    }
  }
  fmt::print("]\n");
}