#ifndef CAFFE_QUANT_HPP_
#define CAFFE_QUANT_HPP_

#include <algorithm>
#include <string>
#include <vector>

namespace caffe {
  enum QuantInferType {
    eNative,
    eFixpoint,
    eFakeQuant
  };

  enum QuantType {
    eFp32,
    eBf16,
    eInt16,
    eInt8,
    eInt4
  };
}  // namespace caffe

#endif  // CAFFE_QUANT_HPP_
