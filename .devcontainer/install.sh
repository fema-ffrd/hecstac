cd ..
git clone https://github.com/fema-ffrd/rasqc.git
cd rasqc/ && git checkout feature/stac-checker

pip install build && python -m build
pip install dist/rasqc-0.0.1rc1-py3-none-any.whl


cd .. && python -m build
pip install dist/hecstac-0.1.0rc3-py3-none-any.whl