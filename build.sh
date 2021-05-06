set -e
pushd /workspace

if [ ! -e build ]; then
  mkdir -p build
fi

pushd build
cmake ../caffe-quant
make -j"$(nproc)" && make install
popd

pushd util/rcnn
make -j"$(nproc)"
popd

popd