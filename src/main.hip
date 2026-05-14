#include <iostream>
#include "halfedge2.hpp"
#include "CLI11.hpp"
#include <hip/hip_runtime.h>

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
    
    // get all outgoing halfedges with he.twin.next loop
    size_t init_he = dev_vert_he[i];
    size_t curr_he = init_he;
    size_t next_he = dev_edges[dev_edges[curr_he + 2] + 3]; // next = curr.twin.next
    
    // for first loop, we need a complete traversal in order to know how much
    // memory to allocate
    int k = 0;
    do {
        k++;
    } while (curr_he != init_he);

    // for the second loop, we can allocate the memory we need and store the 
    // adjacent outgoing halfedges
    size_t neighbors[3];
    curr_he = init_he;
    int j = 0;

    if (i == 0) {
      printf("current he: %zu\n", curr_he);
      printf("next he: %zu\n", dev_edges[curr_he + 2]);
    }
    do {
        neighbors[j] = curr_he;
        curr_he = dev_edges[dev_edges[curr_he + 2] + 3]; // curr = curr.twin.next
        j++;
        printf("j val: %d", j);
    } while (curr_he != init_he);
    
    // for each outgoing halfedge, record the last two halfedges of 
    // the face to the array, find the corresponding lengths, compute 
    // the incident angle of that face, and subtract that from 2 * PI
    // i.e. compute discrete Gaussian curvature via angle defect:
    // K = 2 * PI - sum(angles)
    double angle_def = 2.0 * M_PI;
    for (j = 0; j < 3; j++) {
        // get each edge of the face
        size_t e1 = neighbors[j];
        size_t e2 = dev_edges[e1 + 3]; // e1.next
        size_t e3 = dev_edges[e2 + 3]; // e2.next
        
        double cos_angle = (pow(dev_edge_lens[e1], 2.0) + pow(dev_edge_lens[e3], 2.0) - pow(dev_edge_lens[e2], 2.0)) / (2 * dev_edge_lens[e1] * dev_edge_lens[e3]);
        printf("cos_angle: %f", cos_angle);
        if (cos_angle >= 0.0) {
            cos_angle = __saturatef(cos_angle);
          } else {
            cos_angle = -1.0 * __saturatef(cos_angle);
          }
        angle_def -= acos(cos_angle);
    }

    // writeback the curvature value to the destination array
    dev_dest[i] = angle_def;
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

    // Get GPU-friendly data
    GPUTriangulation gpu_mesh{vertex_count, he_count};
    gpu_mesh.inputVertexData(mesh.getVertices());
    gpu_mesh.inputHalfedgeData(mesh.getHalfedges());

    printArray(gpu_mesh.vertexHalfedges(), vertex_count);

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
    hipHostMalloc(&h_coords, 3*vertex_count*sizeof(double));
    hipHostMalloc(&h_vhalfed, vertex_count*sizeof(size_t));
    hipHostMalloc(&h_edges, 4*he_count*sizeof(size_t));
    hipHostMalloc(&h_vcurv, vertex_count*sizeof(double));
    hipHostMalloc(&h_elens, he_count*sizeof(double));

    // Assign host arrays
    mesh.fillCoordsArray(h_coords);
    mesh.fillEdgeArray(h_edges);
    gpu_mesh.fillArrayData(vertex_count, gpu_mesh.vertexHalfedges(), h_vhalfed);

    // Allocate device (GPU) memory
    hipMalloc(&d_coords, 3*vertex_count*sizeof(double));
    hipMalloc(&h_vhalfed, vertex_count*sizeof(size_t));
    hipMalloc(&d_edges, 4*he_count*sizeof(size_t));
    hipMalloc(&d_vcurv, vertex_count*sizeof(double));
    hipMalloc(&d_elens, he_count*sizeof(double));

    // // Edge length computation // //

    // Memcopy host -> device
    hipMemcpy(d_coords, h_coords, 3*vertex_count*sizeof(double), hipMemcpyDefault);
    hipMemcpy(d_edges, h_edges, 4*he_count*sizeof(size_t), hipMemcpyDefault);
    
    // Call kernel
    int threads_per_block = 256;
    int blocks = (int(he_count) + threads_per_block) / threads_per_block;
    compEdgeLens<<<blocks, threads_per_block>>>(d_edges, d_coords, d_elens, he_count);
    
    hipDeviceSynchronize();

    // Memcopy device -> host
    hipMemcpy(h_elens, d_elens, he_count*sizeof(double), hipMemcpyDefault);

    // Free host and device memory for coords --- the next computation doesn't
    // require it
    hipFreeHost(h_coords);
    hipFree(d_coords);

    // // Vertex curvature computation // //

    hipDeviceReset();

    // Reallocate memory
    // Allocate host memory via HIP
    hipHostMalloc(&h_coords, 3*vertex_count*sizeof(double));
    hipHostMalloc(&h_vhalfed, vertex_count*sizeof(size_t));
    hipHostMalloc(&h_edges, 4*he_count*sizeof(size_t));
    hipHostMalloc(&h_vcurv, vertex_count*sizeof(double));
    hipHostMalloc(&h_elens, he_count*sizeof(double));

    // Assign host arrays
    mesh.fillCoordsArray(h_coords);
    mesh.fillEdgeArray(h_edges);
    gpu_mesh.fillArrayData(vertex_count, gpu_mesh.vertexHalfedges(), h_vhalfed);

    // Allocate device (GPU) memory
    hipMalloc(&d_coords, 3*vertex_count*sizeof(double));
    hipMalloc(&h_vhalfed, vertex_count*sizeof(size_t));
    hipMalloc(&d_edges, 4*he_count*sizeof(size_t));
    hipMalloc(&d_vcurv, vertex_count*sizeof(double));
    hipMalloc(&d_elens, he_count*sizeof(double));

    // Memcopy host -> device
    hipMemcpy(d_edges, h_edges, 4*he_count*sizeof(size_t), hipMemcpyDefault);
    hipMemcpy(d_vhalfed, h_vhalfed, vertex_count*sizeof(size_t), hipMemcpyDefault);

    // Call kernel
    compVertexCurvature<<<blocks, threads_per_block>>>(d_edges, d_vhalfed, d_elens, d_vcurv, vertex_count);

    hipDeviceSynchronize();

    // Memcopy device -> host
    hipMemcpy(h_vcurv, d_vcurv, vertex_count*sizeof(double), hipMemcpyDefault);


    std::cout << "Edge lengths: ";
    for(size_t i = 0; i < he_count; i++) {
        std::cout << h_elens[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Vertex curvatures: ";
    for(size_t i = 0; i < vertex_count; i++) {
        std::cout << h_vcurv[i] << " ";
    }
    std::cout << std::endl;

    hipDeviceReset();

    return 0;
}
