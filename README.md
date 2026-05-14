# Vertex Curvature Computation
This program computes the discrete Gaussian curvature (angle defect) at each vertex. So far, it only considers closed, manifold, and triangular meshes, and computes curvature linearly with standard C++20. 

## Building
First, clone the project using `git clone https://github.com/vaughnzaayer/gpu-curvature-computation.git`. Then, `cd` into the project directory and run the following to build:

```
cmake -B build
```
or to build with optimizations,
```
cmake -B build -DCMAKE_BUILD_TYPE=Release
```
Note that you will need the minimum requirements of C++20 and CMAKE version 3.21 or greater. To compile, run
```
cmake --build build
```

### Building with HIP to use a GPU
First, ensure you have the HIP compiler [installed](https://rocm.docs.amd.com/projects/HIP/en/docs-6.0.0/how_to_guides/install.html), as well as NIVIDA's CUDA or AMD's ROCm, depending on your hardware. In the project's root directory, run
```
hipcc -std=c++20 -I include/ -o hip_main src/main.hip src/halfedge2.cpp
```

Then, run using `./hip_main [filename.obj]`.

To use purely CUDA, compile with this instead:
```
nvcc -std=c++20 -I include/ -o cuda_main src/main.cu src/halfedge2.cpp
```

Then, run using `./cuda_main [filename.obj]`.

## Running
To run the program, use
```
./build/linear_curvature_computation [filename.obj]
```
Without passing a filename, it will default to using `data/cube.obj`. Other options included are `data/bunny.obj` (which has 3300 vertices) and `data/cow.obj` (which has 400,000 vertices).
