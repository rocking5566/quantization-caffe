if [ ! -e build ]; then
  mkdir -p build
fi

pushd build
cmake ../caffe-quant
make -j"$(nproc)" && make install
popd
