#include <iostream>
#include "halfedge2.hpp"
#include "CLI11.hpp"

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

    mesh.computeEdgeLengths();
    std::cout << "Edge lengths: ";
    mesh.printEdgeLengths();

    mesh.computeVertexCurvatures();
    std::cout << "Vertex curvatures: ";
    mesh.printVertexCurvatures();

    return 0;
}