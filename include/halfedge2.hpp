#pragma once

#include <cmath>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <tuple>
#include <algorithm>
#include <limits>
#include <numbers>

struct TupleHash {
    template <class T1, class T2>
    std::size_t operator () (const std::tuple<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(std::get<0>(p));
        auto h2 = std::hash<T2>{}(std::get<1>(p));
        return h1 ^ (h2 << 1); 
    }
};

struct Triangulation;

struct Vertex {
    private:
        Triangulation* tri_;
        size_t id_;
        size_t halfedge_;
        double x_;
        double y_;
        double z_;
    public:
        Vertex(double x, double y, double z);

        Triangulation* getTriangulation();
        void setTriangulation(Triangulation* tri);

        size_t getId();
        void setId(size_t id);

        size_t getHalfedge();
        void setHalfedge(size_t he_id);

        std::tuple<double, double, double> getCoords();
};

struct HalfEdge {
    private:
        Triangulation* tri_;
        size_t id_;
        size_t src_;
        size_t tgt_;
        size_t twin_ = std::numeric_limits<size_t>::max();
        size_t next_ = std::numeric_limits<size_t>::max();
    public:
        HalfEdge(size_t v1, size_t v2);

        Triangulation* getTriangulation();
        void setTriangulation(Triangulation* tri);

        size_t getId();
        void setId(size_t id);

        size_t getSourceVertex();
        void setSourceVertex(size_t src_id);

        size_t getTargetVertex();
        void setTargetVertex(size_t tgt_id);

        size_t getTwinHalfedge();
        void setTwinHalfedge(size_t twin_id);

        size_t getNextHalfedge();
        void setNextHalfedge(size_t next_id);
};

struct Triangulation {
    private:
        std::vector<Vertex> vertices_;
        std::vector<HalfEdge> halfedges_;
        std::unordered_map<std::tuple<size_t, size_t>, size_t, TupleHash> edge_map_;
        std::vector<double> edge_lengths_;
        std::vector<double> vertex_curvatures_;
    public:
        std::vector<Vertex>* getVertices();
        std::vector<HalfEdge>* getHalfedges();

        size_t addNewVertex(double x, double y, double z);
        size_t addNewHalfedge(size_t v1, size_t v2);
        
        void loadFromObj(std::string filepath);
        
        void computeEdgeLengths();
        std::vector<double> getEdgeLengths();
        void printEdgeLengths();

        void computeVertexCurvatures();
        std::vector<double> getVertexCurvatures();
        void printVertexCurvatures();
        double getTotalCurvature();

        size_t getVertexCount();
        size_t getHalfedgeCount();

        Vertex* verticesAsArray();
        HalfEdge* halfedgesAsArray();
        double* edgeLengthsAsArray();
        double* vertCurvaturesAsArray();
};