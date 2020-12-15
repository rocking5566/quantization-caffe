# quantization-caffe
- Just load fp32 caffe model and import calibration table, quantize the tensor after net constructor.
- Choose the precision (int16, int8, int4..bf16 etc) of each op when loadding the model.
- Choose the quantization granularity (perchannel or perlayer) of each tensor when loadding the model.
