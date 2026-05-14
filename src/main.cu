#include <iostream>
#include <algorithm>
#include <array>
#include "halfedge2.hpp"
#include "CLI11.hpp"
#include <cuda_runtime.h>

__device__ __host__
void printArray(const size_t* idx_arr, size_t len) {
  printf("Printing array... \n");
  for (size_t i = 0; i < len; i++) {
    printf("%zu ", idx_arr[i]);
  }
  printf("\n Done! \n");
}

__device__ __host__
void printArray(const double* idx_arr, size_t len) {
  printf("Printing array... \n");
  for (size_t i = 0; i < len; i++) {
    printf("%f ", idx_arr[i]);
  }
  printf("\n Done! \n");
}

__global__ 
void compEdgeLens(const size_t* dev_edges, 
    const double* dev_coords, 
    double* dev_dest, 
    size_t n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        size_t v1_idx = dev_edges[i*4];
        size_t v2_idx = dev_edges[i*4 + 1];
        double dist = sqrt(pow(dev_coords[v1_idx*3] - dev_coords[v2_idx*3],2.0) + pow(dev_coords[v1_idx*3 + 1] - dev_coords[v2_idx*3 + 1],2.0) + pow(dev_coords[v1_idx*3 + 2]-dev_coords[v2_idx*3 + 2],2.0));
        dev_dest[i] = dist;
    }
}

__global__ 
void compVertexCurvature(const size_t* dev_edges,
    const size_t* dev_vert_he, 
    const double* dev_edge_lens, 
    double* dev_dest, 
    size_t n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    double angle_defect = 2 * M_PI;
    
    if (i < n) {

    size_t init_he = dev_vert_he[i];
    size_t curr_he = init_he;
    
    do {
        size_t e1 = curr_he;
        size_t e2 = dev_edges[curr_he + 3];
        size_t e3 = dev_edges[e2 + 3];


        double len_e1 = dev_edge_lens[e1 / 4];
        double len_e2 = dev_edge_lens[e2 / 4];
        double len_e3 = dev_edge_lens[e3 / 4];

        double cos_angle = (pow(len_e1, 2.0) + pow(len_e3, 2.0) - pow(len_e2, 2.0)) / (2 * len_e1 * len_e3);
        
            
        if (cos_angle >= 0.0) {
            cos_angle = __saturatef(cos_angle);
        } else {
            cos_angle = -__saturatef(-cos_angle);
        }

        angle_defect -= acos(cos_angle);
        auto twin = dev_edges[curr_he + 2];
        auto twin_next = dev_edges[twin + 3];
        curr_he = twin_next;
      
    } while (curr_he != init_he);
    
    // writeback the curvature value to the destination array
    dev_dest[i] = angle_defect;
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

    size_t vertex_count = vertices->size();
    size_t he_count = halfedges->size();

    // Get GPU-friendly data
    GPUTriangulation gpu_mesh{vertex_count, he_count};
    gpu_mesh.inputVertexData(mesh.getVertices());
    gpu_mesh.inputHalfedgeData(mesh.getHalfedges());

    // Pointers for host memory
    double* h_coords = nullptr;
    size_t* h_vhalfed = nullptr;
    size_t* h_edges = nullptr;
    double* h_vcurv = nullptr;
    double* h_elens = nullptr;

    // Pointers for device memory
    double* d_coords = nullptr;
    size_t* d_vhalfed = nullptr;
    size_t* d_edges = nullptr;
    double* d_vcurv = nullptr;
    double* d_elens = nullptr;

    // Allocate host memory via HIP
    cudaMallocHost(&h_coords, 3*vertex_count*sizeof(double));
    cudaMallocHost(&h_vhalfed, vertex_count*sizeof(size_t));
    cudaMallocHost(&h_edges, 4*he_count*sizeof(size_t));
    cudaMallocHost(&h_vcurv, vertex_count*sizeof(double));
    cudaMallocHost(&h_elens, he_count*sizeof(double));

    // Assign host arrays
    gpu_mesh.fillArrayData(vertex_count * 3, gpu_mesh.vertexCoordinates(), h_coords);
    gpu_mesh.fillArrayData(he_count * 4, gpu_mesh.halfedges(), h_edges);
    gpu_mesh.fillArrayData(vertex_count, gpu_mesh.vertexHalfedges(), h_vhalfed);

    // Allocate device (GPU) memory
    cudaMalloc(&d_coords, 3*vertex_count*sizeof(double));
    cudaMalloc(&d_vhalfed, vertex_count*sizeof(size_t));
    cudaMalloc(&d_edges, 4*he_count*sizeof(size_t));
    cudaMalloc(&d_vcurv, vertex_count*sizeof(double));
    cudaMalloc(&d_elens, he_count*sizeof(double));

    // // Edge length computation // //

    // Memcopy host -> device
    cudaMemcpy(d_coords, h_coords, 3*vertex_count*sizeof(double), cudaMemcpyDefault);
    cudaMemcpy(d_edges, h_edges, 4*he_count*sizeof(size_t), cudaMemcpyDefault);
    
    // Call kernel
    int threads_per_block = 256;
    int blocks = (int(he_count) + threads_per_block) / threads_per_block;
    compEdgeLens<<<blocks, threads_per_block>>>(d_edges, d_coords, d_elens, he_count);
    
    cudaDeviceSynchronize();

    // Memcopy device -> host
    cudaMemcpy(h_elens, d_elens, he_count*sizeof(double), cudaMemcpyDefault);

    double edge_lengths[he_count];
    cudaMemcpy(edge_lengths, h_elens, he_count*sizeof(double), cudaMemcpyDefault);

    // // Vertex curvature computation // //

    cudaDeviceReset();

    // Reallocate memory
    // Allocate host memory via HIP
    cudaMallocHost(&h_coords, 3*vertex_count*sizeof(double));
    cudaMallocHost(&h_vhalfed, vertex_count*sizeof(size_t));
    cudaMallocHost(&h_edges, 4*he_count*sizeof(size_t));
    cudaMallocHost(&h_vcurv, vertex_count*sizeof(double));
    cudaMallocHost(&h_elens, he_count*sizeof(double));

    // Assign host arrays
    gpu_mesh.fillArrayData(vertex_count * 3, gpu_mesh.vertexCoordinates(), h_coords);
    gpu_mesh.fillArrayData(he_count * 4, gpu_mesh.halfedges(), h_edges);
    gpu_mesh.fillArrayData(vertex_count, gpu_mesh.vertexHalfedges(), h_vhalfed);

    // Allocate device (GPU) memory
    cudaMalloc(&d_coords, 3*vertex_count*sizeof(double));
    cudaMalloc(&d_vhalfed, vertex_count*sizeof(size_t));
    cudaMalloc(&d_edges, 4*he_count*sizeof(size_t));
    cudaMalloc(&d_vcurv, vertex_count*sizeof(double));
    cudaMalloc(&d_elens, he_count*sizeof(double));

    // Memcopy host -> device
    cudaMemcpy(d_edges, h_edges, 4*he_count*sizeof(size_t), cudaMemcpyDefault);
    cudaMemcpy(d_vhalfed, h_vhalfed, vertex_count*sizeof(size_t), cudaMemcpyDefault);
    cudaMemcpy(d_elens, edge_lengths, he_count*sizeof(double), cudaMemcpyDefault);

    

    // Call kernel
    compVertexCurvature<<<blocks, threads_per_block>>>(d_edges, d_vhalfed, d_elens, d_vcurv, vertex_count);

    cudaDeviceSynchronize();

    // Memcopy device -> host
    cudaMemcpy(h_vcurv, d_vcurv, vertex_count*sizeof(double), cudaMemcpyDefault);
    cudaGetLastError();

    std::cout << "Edge lengths: ";
    for(size_t i = 0; i < he_count; i++) {
        std::cout << edge_lengths[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Vertex curvatures: ";
    for(size_t i = 0; i < vertex_count; i++) {
        std::cout << h_vcurv[i] << " ";
    }
    std::cout << std::endl;

    // Validating using the Gauss-Bonnet Theorem
    int f = int(mesh.getFaceCount());
    int v = int(mesh.getVertexCount());
    int e = int(mesh.getHalfedgeCount()) / 2;
    int euler_char = v - e + f;

    double total_curvature = 0;

    for (size_t i = 0; i < vertex_count; i++) {
        total_curvature += h_vcurv[i];
    }

    auto diff = std::abs(2 * M_PI * double(euler_char) - total_curvature);
    if (diff < 1e-5) {
        printf("Gauss-Bonnet Theorem valid.\n");
    } else {
        printf("Gauss-Bonnet Theorem invalid.\n");
    }



    cudaDeviceReset();

    return 0;
}
