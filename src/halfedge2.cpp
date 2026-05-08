#include "halfedge2.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"


///////////////////////////////////////////////////////////
//                Triangulation Methods                  //
///////////////////////////////////////////////////////////

std::vector<Vertex>* Triangulation::getVertices() {
    return &vertices_;
}

std::vector<HalfEdge>* Triangulation::getHalfedges() {
    return &halfedges_;
}

size_t Triangulation::addNewVertex(double x, double y, double z) {
    Vertex vert = Vertex{x, y, z};
    vert.setId(vertices_.size());
    vert.setTriangulation(this);
    vertices_.push_back(vert);
    return vert.getId();
}

size_t Triangulation::addNewHalfedge(size_t v1, size_t v2) {
    std::tuple<size_t, size_t> key(std::min(v1, v2), std::max(v1,v2));
    HalfEdge he = HalfEdge{v1, v2};
    auto id = halfedges_.size();
    he.setId(id);
    he.setTriangulation(this);

    if (edge_map_.contains(key)) {
        auto twin = edge_map_[key];
        he.setTwinHalfedge(twin);
        halfedges_[twin].setTwinHalfedge(id);
    } else {
        edge_map_[key] = id;
    }

    halfedges_.push_back(he);
    vertices_[v1].setHalfedge(id);

    return id;
}

void Triangulation::loadFromObj(std::string inputfile) {
    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(inputfile)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        exit(1);
    }
    
    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();

    std::unordered_map<size_t, size_t> tinyobjToLocalIdx;

    // iterate over all shapes
    for (size_t s = 0; s < shapes.size(); s++) {
        size_t index_offset = 0;

        // iterate over all faces in a shape
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            // assuming a triangle mesh, this should be 3
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);
            std::vector<size_t> verts;
            // iterate over all vertices in a face
            for (size_t v = 0; v < fv; v++) {
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                size_t tinyobjIdx = size_t(idx.vertex_index);
                if (!tinyobjToLocalIdx.contains(tinyobjIdx)) {
                    tinyobj::real_t vx = attrib.vertices[3*size_t(idx.vertex_index) + 0];
                    tinyobj::real_t vy = attrib.vertices[3*size_t(idx.vertex_index) + 1];
                    tinyobj::real_t vz = attrib.vertices[3*size_t(idx.vertex_index) + 2];
                    tinyobjToLocalIdx[tinyobjIdx] = addNewVertex(vx, vy, vz);
                }
                verts.push_back(tinyobjToLocalIdx[tinyobjIdx]);
            }
            index_offset += fv;
            std::vector<size_t> face_edges;
            for (size_t i = 0; i < verts.size(); i++) {
                size_t he_id;
                if (i + 1 < verts.size()) {
                    he_id = addNewHalfedge(verts[i], verts[i+1]);
                } else {
                    he_id = addNewHalfedge(verts[i], verts[0]);
                }
                face_edges.push_back(he_id);
            }

            for (size_t i = 0; i < face_edges.size(); i++) {
                if (i + 1 < verts.size()) {
                    halfedges_[face_edges[i]].setNextHalfedge(face_edges[i+1]);
                } else {
                    halfedges_[face_edges[i]].setNextHalfedge(face_edges[0]);
                }
            }
        }
    }
}

void Triangulation::computeEdgeLengths() {
    for (size_t e = 0; e < halfedges_.size(); e++) {
        auto src_coords = vertices_[halfedges_[e].getSourceVertex()].getCoords();
        auto tgt_coords = vertices_[halfedges_[e].getTargetVertex()].getCoords();
        double len = std::sqrt(
            std::pow(std::get<0>(src_coords) - std::get<0>(tgt_coords), 2) 
            + std::pow(std::get<1>(src_coords) - std::get<1>(tgt_coords), 2) 
            + std::pow(std::get<2>(src_coords) - std::get<2>(tgt_coords), 2));
        edge_lengths_.push_back(len);
    }
}

std::vector<double> Triangulation::getEdgeLengths() {
    return edge_lengths_;
}

void Triangulation::printEdgeLengths() {
    for (size_t e = 0; e < edge_lengths_.size(); e++) {
        std::cout << edge_lengths_[e] << " ";
    }
    std::cout << std::endl;
}

void Triangulation::computeVertexCurvatures() {
    for (size_t v = 0; v < vertices_.size(); v++) {
        std::vector<size_t> outgoing_edges;
        std::vector<double> face_edge_lens;
        std::vector<double> incident_angles;
        // collect all outgoing edges for vertex v
        auto curr_he = vertices_[v].getHalfedge();
        do {
            outgoing_edges.push_back(curr_he);
            curr_he = halfedges_[halfedges_[curr_he].getTwinHalfedge()].getNextHalfedge();
        } while (curr_he != vertices_[v].getHalfedge());

        // traverse each outgoing halfedge to get the edge lengths for each 
        // neighboring face (can be done in parallel?)
        for (size_t f = 0; f < outgoing_edges.size(); f++) {
            auto curr_he = outgoing_edges[f];
            auto start_he = curr_he;
            do {
                face_edge_lens.push_back(edge_lengths_[curr_he]);
                curr_he = halfedges_[curr_he].getNextHalfedge();
            } while(curr_he != start_he);
        }

        // (assuming a triangle mesh) use law of cosines to compute interior angles
        for (size_t e = 0; e < face_edge_lens.size(); e += 3) {
            auto cos_angle = std::clamp((std::pow(face_edge_lens[e], 2) + std::pow(face_edge_lens[e+2], 2) - std::pow(face_edge_lens[e+1], 2)) / (2 * face_edge_lens[e] * face_edge_lens[e+2]), -1.0, 1.0);
            incident_angles.push_back(std::acos(cos_angle));
        }

        // compute angle defect using incident angles (discrete Gaussian curvature)
        double ang_sum = 0;
        for (double a : incident_angles) {
            ang_sum += a;
        }
        vertex_curvatures_.push_back(2 * std::numbers::pi - ang_sum);
    }
}

std::vector<double> Triangulation::getVertexCurvatures() {
    return vertex_curvatures_;
}

void Triangulation::printVertexCurvatures() {
    for (size_t v = 0; v < vertex_curvatures_.size(); v++) {
        std::cout << vertex_curvatures_[v] << " ";
    }
    std::cout << std::endl;
}

double Triangulation::getTotalCurvature() {
    auto total = 0;
    for (size_t v = 0; v < vertex_curvatures_.size(); v++) {
        total += vertex_curvatures_[v];
    }
    return total;
}

size_t Triangulation::getVertexCount() {
    return vertices_.size();
}

size_t Triangulation::getHalfedgeCount() {
    return halfedges_.size();
}

Vertex* Triangulation::verticesAsArray() {
    return vertices_.data();
}

HalfEdge* Triangulation::halfedgesAsArray() {
    return halfedges_.data();
}

double* Triangulation::edgeLengthsAsArray() {
    return edge_lengths_.data();
}

double* Triangulation::vertCurvaturesAsArray() {
    return vertex_curvatures_.data();
}

double* Triangulation::coordsAsArray() {
    std::vector<double> coords_array;
    for (Vertex v : vertices_) {
        coords_array.push_back(v.x());
        coords_array.push_back(v.y());
        coords_array.push_back(v.z());
    }
    return coords_array.data();
}

size_t* Triangulation::edgeSrcAndTgtAsArray() {
    std::vector<size_t> idx_array;
    for (HalfEdge e : halfedges_) {
        idx_array.push_back(e.getSourceVertex());
        idx_array.push_back(e.getTargetVertex());
    }
    return idx_array.data();
}


///////////////////////////////////////////////////////////
//                  Vertex Methods                       //
///////////////////////////////////////////////////////////

Vertex::Vertex(double x, double y, double z) {
    x_ = x;
    y_ = y;
    z_ = z;
}

double Vertex::x() {
    return x_;
}

double Vertex::y() {
    return y_;
}

double Vertex::z() {
    return z_;
}

Triangulation* Vertex::getTriangulation() {
    return tri_;
}

void Vertex::setTriangulation(Triangulation* tri) {
    tri_ = tri;
}

size_t Vertex::getId() {
    return id_;
}

void Vertex::setId(size_t id) {
    id_ = id;
}

size_t Vertex::getHalfedge() {
    return halfedge_;
}

void Vertex::setHalfedge(size_t he_id) {
    halfedge_ = he_id;
}

std::tuple<double, double, double> Vertex::getCoords() {
    return std::tuple<double, double, double>(x_, y_, z_);
}


///////////////////////////////////////////////////////////
//                   Halfedge Methods                    //
///////////////////////////////////////////////////////////

HalfEdge::HalfEdge(size_t v1, size_t v2) {
    src_ = v1;
    tgt_ = v2;
}

Triangulation* HalfEdge::getTriangulation() {
    return tri_;
}
void HalfEdge::setTriangulation(Triangulation* tri) {
    tri_ = tri;
}

size_t HalfEdge::getId() {
    return id_;
}
void HalfEdge::setId(size_t id) {
    id_ = id;
}

size_t HalfEdge::getSourceVertex() {
    return src_;
}
void HalfEdge::setSourceVertex(size_t src_id) {
    src_ = src_id;
}

size_t HalfEdge::getTargetVertex() {
    return tgt_;
}
void HalfEdge::setTargetVertex(size_t tgt_id) {
    tgt_ = tgt_id;
}

size_t HalfEdge::getTwinHalfedge() {
    return twin_;
}
void HalfEdge::setTwinHalfedge(size_t twin_id) {
    twin_ = twin_id;
}

size_t HalfEdge::getNextHalfedge() {
    return next_;
}

void HalfEdge::setNextHalfedge(size_t next_id) {
    next_ = next_id;
}