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

## Running
To run the program, use
```
./build/linear_curvature_computation [filename]
```
Without passing a filename, it will default to using `data/cube.obj`. Other options included are `data/bunny.obj` (which has 3300 vertices) and `data/cow.obj` (which has 400,000 vertices).