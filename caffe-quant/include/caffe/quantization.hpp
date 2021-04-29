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

  enum BlobQuantType {
    eFp32,
    eBf16,
    eInt16,
    eInt8,
    eInt4
  };

  inline const char* BlobQuantTypeToString(BlobQuantType qtype) {
    if (qtype == eFp32)
      return "float32";
    else if (qtype == eBf16)
      return "bfloat16";
    else if (qtype == eInt16)
      return "int16";
    else if (qtype == eInt8)
      return "int8";
    else if (qtype == eInt4)
      return "int4";
    else
      return "Unknown type";
  }
}  // namespace caffe

#endif  // CAFFE_QUANT_HPP_
