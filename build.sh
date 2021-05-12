set -e
pushd /workspace

if [ ! -e build ]; then
  mkdir -p build
fi

pushd build
if [ ! -e caffe ]; then
  mkdir -p caffe
fi

pushd caffe
cmake ../../caffe-quant
make -j"$(nproc)" && make install
popd

if [ ! -e calibration_tool ]; then
  mkdir -p calibration_tool
fi

pushd calibration_tool
cmake ../../calibration_tool
make -j"$(nproc)"
popd

# build
popd


pushd util/rcnn
make -j"$(nproc)"
popd