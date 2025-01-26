rm -rf build
sphinx-apidoc -o source ../hecstac
sphinx-build -M html source build