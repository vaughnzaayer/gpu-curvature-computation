#include <iostream>
#include "halfedge2.hpp"
#include "CLI11.hpp"
#include <hip/hip_runtime.h>

__global__ void compEdgeLens(const size_t* dev_edges, const double* dev_coords, double* dev_dest, size_t n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        size_t v1_idx = dev_edges[i*2];
        size_t v2_idx = dev_edges[i*2 + 1];
        double dist = sqrt(pow(dev_coords[v1_idx*3] - dev_coords[v2_idx*3],2.0) + pow(dev_coords[v1_idx*3 + 1] - dev_coords[v2_idx*3 + 1],2.0) + pow(dev_coords[v1_idx*3 + 2]-dev_coords[v2_idx*3 + 2],2.0));
        dev_dest[i] = dist;
      }
  }

int main(int argc, char** argv) {
    CLI::App app{"Computes the curvature at every vertex of a mesh. Assumes the mesh is closed and manifold."};
    argv = app.ensure_utf8(argv);

    std::string filename = "data/cube.obj";
    app.add_option("filename", filename, "Path to the input .obj file. Included is data/cube.obj, data/bunny.obj, and data/cow.obj. Defaults to data/cube.obj.");

    CLI11_PARSE(app, argc, argv);
    
    Triangulation mesh;
    
    std::cout << "Loading mesh..." << std::endl;
    mesh.loadFromObj(filename);
    
    auto vertices = mesh.getVertices();
    auto halfedges = mesh.getHalfedges();
    
    std::cout << "Vertices loaded: " << vertices->size() << std::endl;
    std::cout << "Half-edges created: " << halfedges->size() << std::endl;

    size_t boundaries = 0;
    for (auto& he : *halfedges) {
        if (he.getTwinHalfedge() == std::numeric_limits<size_t>::max()) {
            boundaries++;
        }
    }
    
    std::cout << "Boundary half-edges (no twins): " << boundaries << std::endl;

    std::cout << "Computing Using CPU:" << std::endl;

    mesh.computeEdgeLengths();
    std::cout << "Edge lengths: ";
    mesh.printEdgeLengths();

    mesh.computeVertexCurvatures();
    std::cout << "Vertex curvatures: ";
    mesh.printVertexCurvatures();

    std::cout << "Computing Using GPU:" << std::endl;

    size_t vertex_count = mesh.getVertexCount();
    size_t he_count = mesh.getHalfedgeCount();


    // Pointers for host memory
    double* h_vs = nullptr;
    size_t* h_es = nullptr;
    double* h_vks = nullptr;
    double* h_els = nullptr;

    // Pointers for device memory
    double* d_vs = nullptr;
    size_t* d_es = nullptr;
    double* d_vks = nullptr;
    double* d_els = nullptr;

    // Allocate host memory via HIP
    hipHostMalloc(&h_vs, 3*vertex_count*sizeof(double));
    hipHostMalloc(&h_es, 2*he_count*sizeof(size_t));
    hipHostMalloc(&h_vks, vertex_count*sizeof(double));
    hipHostMalloc(&h_els, he_count*sizeof(double));

    // Assign host arrays
    mesh.fillCoordsArray(h_vs);
    mesh.fillEdgeVtxArray(h_es);

    // Allocate device (GPU) memory
    hipMalloc(&d_vs, 3*vertex_count*sizeof(double));
    hipMalloc(&d_es, 2*he_count*sizeof(size_t));
    hipMalloc(&d_vks, vertex_count*sizeof(double));
    hipMalloc(&d_els, he_count*sizeof(double));

    // Memcopy host -> device
    hipMemcpy(d_vs, h_vs, 3*vertex_count*sizeof(double), hipMemcpyDefault);
    hipMemcpy(d_es, h_es, 2*he_count*sizeof(size_t), hipMemcpyDefault);
    
    // Call kernel
    int threads_per_block = 256;
    int blocks = (int(he_count) + threads_per_block) / threads_per_block;
    compEdgeLens<<<blocks, threads_per_block>>>(d_es, d_vs, d_els, he_count);
    
    hipDeviceSynchronize();

    // Memcopy device -> host
    hipMemcpy(h_els, d_els, he_count*sizeof(double), hipMemcpyDefault);

    std::cout << "Edge lengths: ";
    for(size_t i = 0; i < he_count; i++) {
        std::cout << h_els[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
