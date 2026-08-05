# mario.mocet
We are trying to apply the Mocet framework to the CNeuromod Mario All Stars dataset

To install eyerec-python on elm:

```bash
cd .venv/lib/python3.10/site-packages/
git clone --recursive https://github.com/tcsantini/eyerec-python.git
cd eyerec-python/lib/cpp

# use miniconda in your home folder to $HOME/miniconda3/bin/conda install -c conda-forge opencv -y
# in the future ask Basile, but for now he is in vacation.
g++ -O2 -Wall -c -fPIC \
  -I$HOME/miniconda3/include/opencv4 \
  -I./src -I./src/common -I./include \
  src/common/ocv_utils.cpp \
  src/pupil/detection/PupilDetectionMethod.cpp \
  src/pupil/detection/PuRe/PuRe.cpp \
  src/pupil/tracking/PupilTrackingMethod.cpp \
  src/pupil/tracking/PuReST/PuReST.cpp 
ar rcs libeyerec_cpp.a *.o
cd ../..
pip install cython
pip install numpy
cython --cplus -3 eyerec/_eyerec.pyx
pip install -e .

```

