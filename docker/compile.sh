cd softgym/PyFlex/bindings
rm -rf build
mkdir build
cd build
cmake -DPYBIND11_PYTHON_VERSION=3.6 ..
make -j
