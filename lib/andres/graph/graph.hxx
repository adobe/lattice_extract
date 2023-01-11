#pragma once
#ifndef ANDRES_GRAPH_HXX
#define ANDRES_GRAPH_HXX

#include <cassert>
#include <cstddef>
#include <iterator> // std::random_access_iterator
#include <vector>
#include <set> 
#include <iostream>
#include <utility> // std::pair
#include <algorithm> // std::fill

#include "adjacency.hxx"
#include "subgraph.hxx"
#include "visitor.hxx"
#include "detail/graph.hxx"

/// The public API.
namespace andres {

/// Graphs and graph algorithms.
namespace graph {

/// Undirected graph, implemented as an adjacency list.
template<typename VISITOR = IdleGraphVisitor<unsigned long> >
class Graph {
public: 
    typedef VISITOR Visitor;
    typedef detail::VertexIterator VertexIterator;
    typedef detail::EdgeIterator EdgeIterator;
    typedef detail::Adjacencies::const_iterator AdjacencyIterator;    
    typedef typename AdjacencyIterator::value_type AdjacencyType;

    // construction
    Graph(const Visitor& = Visitor());
    Graph(const unsigned long, const Visitor& = Visitor());
    void assign(const Visitor& = Visitor());
    void assign(const unsigned long, const Visitor& = Visitor());
    void reserveVertices(const unsigned long);
    void reserveEdges(const unsigned long);

    // iterator access (compatible with Digraph)
    VertexIterator verticesFromVertexBegin(const unsigned long) const;
    VertexIterator verticesFromVertexEnd(const unsigned long) const;
    VertexIterator verticesToVertexBegin(const unsigned long) const;
    VertexIterator verticesToVertexEnd(const unsigned long) const;
    EdgeIterator edgesFromVertexBegin(const unsigned long) const;
    EdgeIterator edgesFromVertexEnd(const unsigned long) const;
    EdgeIterator edgesToVertexBegin(const unsigned long) const;
    EdgeIterator edgesToVertexEnd(const unsigned long) const;
    AdjacencyIterator adjacenciesFromVertexBegin(const unsigned long) const;
    AdjacencyIterator adjacenciesFromVertexEnd(const unsigned long) const;
    AdjacencyIterator adjacenciesToVertexBegin(const unsigned long) const;
    AdjacencyIterator adjacenciesToVertexEnd(const unsigned long) const;

    // access (compatible with Digraph)
    unsigned long numberOfVertices() const;
    unsigned long numberOfEdges() const;
    unsigned long numberOfEdgesFromVertex(const unsigned long) const;
    unsigned long numberOfEdgesToVertex(const unsigned long) const;
    unsigned long vertexOfEdge(const unsigned long, const unsigned long) const;
    unsigned long edgeFromVertex(const unsigned long, const unsigned long) const;
    unsigned long edgeToVertex(const unsigned long, const unsigned long) const;
    unsigned long vertexFromVertex(const unsigned long, const unsigned long) const;
    unsigned long vertexToVertex(const unsigned long, const unsigned long) const;
    const AdjacencyType& adjacencyFromVertex(const unsigned long, const unsigned long) const;
    const AdjacencyType& adjacencyToVertex(const unsigned long, const unsigned long) const;
    std::pair<bool, unsigned long> findEdge(const unsigned long, const unsigned long) const;
    bool multipleEdgesEnabled() const;
    void getVerticesOfEdge(const unsigned long edgeIndex, unsigned long &v1, unsigned long &v2);

    // manipulation
    void eraseVertex(const unsigned long);
    void eraseEdge(const unsigned long);
    unsigned long insertVertex();
    unsigned long insertVertices(const unsigned long);
    unsigned long insertEdge(const unsigned long, const unsigned long);
    unsigned long insertSingleEdge(const unsigned long, const unsigned long);
    bool& multipleEdgesEnabled();

private:
    typedef detail::Adjacencies Vertex;
    typedef detail::Edge<false> Edge;

    void insertAdjacenciesForEdge(const unsigned long);
    void eraseAdjacenciesForEdge(const unsigned long);

    std::vector<Vertex> vertices_;
    std::vector<Edge> edges_;
    bool multipleEdgesEnabled_;
    Visitor visitor_;
};

/// Construct an undirected graph.
///
/// \param visitor Visitor to follow changes of integer indices of vertices and edges.
///
template<typename VISITOR>
inline 
Graph<VISITOR>::Graph(
    const Visitor& visitor
)
:   vertices_(),
    edges_(),
    multipleEdgesEnabled_(false),
    visitor_(visitor)
{}

/// Construct an undirected graph with an initial number of vertices.
///
/// \param numberOfVertices Number of vertices.
/// \param visitor Visitor to follow changes of integer indices of vertices and edges.
///
template<typename VISITOR>
inline 
Graph<VISITOR>::Graph(
    const unsigned long numberOfVertices,
    const Visitor& visitor
)
:   vertices_(numberOfVertices),
    edges_(),
    multipleEdgesEnabled_(false),
    visitor_(visitor)
{
    visitor_.insertVertices(0, numberOfVertices);
}

/// Clear an undirected graph.
///
/// \param visitor Visitor to follow changes of integer indices of vertices and edges.
///
template<typename VISITOR>
inline void
Graph<VISITOR>::assign(
    const Visitor& visitor
) {
    vertices_.clear();
    edges_.clear();
    multipleEdgesEnabled_ = false;
    visitor_ = visitor;
}

/// Clear an undirected graph with an initial number of vertices.
///
/// \param numberOfVertices Number of vertices.
/// \param visitor Visitor to follow changes of integer indices of vertices and edges.
///
template<typename VISITOR>
inline void
Graph<VISITOR>::assign(
    const unsigned long numberOfVertices,
    const Visitor& visitor
) {
    vertices_.resize(numberOfVertices);
    std::fill(vertices_.begin(), vertices_.end(), Vertex());
    edges_.clear();
    multipleEdgesEnabled_ = false;
    visitor_ = visitor;
    visitor_.insertVertices(0, numberOfVertices);
}
    
/// Get the number of vertices.
///
template<typename VISITOR>
inline unsigned long
Graph<VISITOR>::numberOfVertices() const { 
    return vertices_.size(); 
}

/// Get the number of edges.
///
template<typename VISITOR>
inline unsigned long
Graph<VISITOR>::numberOfEdges() const { 
    return edges_.size(); 
}

/// Get the number of edges that originate from a given vertex.
///
/// \param vertex Integer index of a vertex.
///
/// \sa edgeFromVertex()
///
template<typename VISITOR>
inline unsigned long
Graph<VISITOR>::numberOfEdgesFromVertex(
    const unsigned long vertex
) const { 
    return vertices_[vertex].size();
}

/// Get the number of edges that are incident to a given vertex.
///
/// \param vertex Integer index of a vertex.
///
/// \sa edgeToVertex()
///
template<typename VISITOR>
inline unsigned long
Graph<VISITOR>::numberOfEdgesToVertex(
    const unsigned long vertex
) const { 
    return vertices_[vertex].size();
}

/// Get the integer index of a vertex of an edge.
///
/// \param edge Integer index of an edge.
/// \param j Number of the vertex in the edge; either 0 or 1.
///
template<typename VISITOR>
inline unsigned long
Graph<VISITOR>::vertexOfEdge(
    const unsigned long edge,
    const unsigned long j
) const {
    assert(j < 2);

    return edges_[edge][j];
}

/// Get the integer index of an edge that originates from a given vertex.
///
/// \param vertex Integer index of a vertex.
/// \param j Number of the edge; between 0 and numberOfEdgesFromVertex(vertex) - 1.
///
/// \sa numberOfEdgesFromVertex()
///
template<typename VISITOR>
inline unsigned long
Graph<VISITOR>::edgeFromVertex(
    const unsigned long vertex,
    const unsigned long j
) const {
    return vertices_[vertex][j].edge();
}

/// Get the integer index of an edge that is incident to a given vertex.
///
/// \param vertex Integer index of a vertex.
/// \param j Number of the edge; between 0 and numberOfEdgesFromVertex(vertex) - 1.
///
/// \sa numberOfEdgesToVertex()
///
template<typename VISITOR>
inline unsigned long
Graph<VISITOR>::edgeToVertex(
    const unsigned long vertex,
    const unsigned long j
) const {
    return vertices_[vertex][j].edge();
}

/// Get the integer index of a vertex reachable from a given vertex via a single edge.
///
/// \param vertex Integer index of a vertex.
/// \param j Number of the vertex; between 0 and numberOfEdgesFromVertex(vertex) - 1.
///
/// \sa numberOfEdgesFromVertex() 
///
template<typename VISITOR>
inline unsigned long
Graph<VISITOR>::vertexFromVertex(
    const unsigned long vertex,
    const unsigned long j
) const {
    return vertices_[vertex][j].vertex();
}

/// Get the integer index of a vertex from which a given vertex is reachable via a single edge.
///
/// \param vertex Integer index of a vertex.
/// \param j Number of the vertex; between 0 and numberOfEdgesFromVertex(vertex) - 1.
///
/// \sa numberOfEdgesFromVertex() 
///
template<typename VISITOR>
inline unsigned long
Graph<VISITOR>::vertexToVertex(
    const unsigned long vertex,
    const unsigned long j
) const {
    return vertices_[vertex][j].vertex();
}

/// Insert an additional vertex.
///
/// \return Integer index of the newly inserted vertex.
///
/// \sa insertVertices()
///
template<typename VISITOR>
inline unsigned long
Graph<VISITOR>::insertVertex() {
    vertices_.push_back(Vertex());
    visitor_.insertVertex(vertices_.size() - 1);
    return vertices_.size() - 1;
}

/// Insert additional vertices.
///
/// \param number Number of new vertices to be inserted.
/// \return Integer index of the first newly inserted vertex.
///
/// \sa insertVertex()
///
template<typename VISITOR>
inline unsigned long
Graph<VISITOR>::insertVertices(
    const unsigned long number
) {
    unsigned long position = vertices_.size();
    vertices_.insert(vertices_.end(), number, Vertex());
    visitor_.insertVertices(position, number);
    return position;
}

/// Insert an additional edge.
///
/// \param vertexIndex0 Integer index of the first vertex in the edge.
/// \param vertexIndex1 Integer index of the second vertex in the edge.
/// \return Integer index of the newly inserted edge.
/// 
template<typename VISITOR>
inline unsigned long
Graph<VISITOR>::insertEdge(
    const unsigned long vertexIndex0,
    const unsigned long vertexIndex1
) {
    assert(vertexIndex0 < numberOfVertices()); 
    assert(vertexIndex1 < numberOfVertices()); 
    
    if(multipleEdgesEnabled()) {
insertEdgeMark:
        edges_.push_back(Edge(vertexIndex0, vertexIndex1));
        unsigned long edgeIndex = edges_.size() - 1;
        insertAdjacenciesForEdge(edgeIndex);
        visitor_.insertEdge(edgeIndex);  
        return edgeIndex;
    }
    else {
        std::pair<bool, unsigned long> p = findEdge(vertexIndex0, vertexIndex1);
        if(p.first) { // edge already exists
            return p.second;
        }
        else {
            goto insertEdgeMark;
        }
    }
}

//this is a custom insert edge function which doesn't check
//if an edge already exists since during graph construction
//we ensure an edge is added only once
template<typename VISITOR>
inline unsigned long
Graph<VISITOR>::insertSingleEdge(
                               const unsigned long vertexIndex0,
                               const unsigned long vertexIndex1
                               ) {
    edges_.push_back(Edge(vertexIndex0, vertexIndex1));
    std::size_t edgeIndex = edges_.size() - 1;
    insertAdjacenciesForEdge(edgeIndex);
    visitor_.insertEdge(edgeIndex);
    return edgeIndex;
}
    
/// Erase a vertex and all edges connecting this vertex.
///
/// \param vertexIndex Integer index of the vertex to be erased.
/// 
template<typename VISITOR>
void 
Graph<VISITOR>::eraseVertex(
    const unsigned long vertexIndex
) {
    assert(vertexIndex < numberOfVertices()); 

    // erase all edges connected to the vertex
    while(vertices_[vertexIndex].size() != 0) {
        eraseEdge(vertices_[vertexIndex].begin()->edge());
    }

    if(vertexIndex == numberOfVertices()-1) { // if the last vertex is to be erased        
        vertices_.pop_back(); // erase vertex
        visitor_.eraseVertex(vertexIndex);
    }
    else { // if a vertex is to be erased which is not the last vertex
        // move last vertex to the free position:

        // collect indices of edges affected by the move
        unsigned long movingVertexIndex = numberOfVertices() - 1;
        std::set<unsigned long> affectedEdgeIndices;
        for(Vertex::const_iterator it = vertices_[movingVertexIndex].begin();
        it != vertices_[movingVertexIndex].end(); ++it) {
            affectedEdgeIndices.insert(it->edge());
        }
        
        // for all affected edges:
        for(std::set<unsigned long>::const_iterator it = affectedEdgeIndices.begin();
        it != affectedEdgeIndices.end(); ++it) { 
            // remove adjacencies
            eraseAdjacenciesForEdge(*it);

            // adapt vertex labels
            for(unsigned long j=0; j<2; ++j) {
                if(edges_[*it][j] == movingVertexIndex) {
                    edges_[*it][j] = vertexIndex;
                }
            }
            // if(!(edges_[*it].directedness()) && edges_[*it][0] > edges_[*it][1]) {
            if(edges_[*it][0] > edges_[*it][1]) {
                std::swap(edges_[*it][0], edges_[*it][1]);
            }
        }

        // move vertex
        vertices_[vertexIndex] = vertices_[movingVertexIndex]; // copy
        vertices_.pop_back(); // erase

        // insert adjacencies for edges of moved vertex
        for(std::set<unsigned long>::const_iterator it = affectedEdgeIndices.begin();
        it != affectedEdgeIndices.end(); ++it) { 
            insertAdjacenciesForEdge(*it);
        }

        visitor_.eraseVertex(vertexIndex);
        visitor_.relabelVertex(movingVertexIndex, vertexIndex);
    }
}

/// Erase an edge.
///
/// \param edgeIndex Integer index of the edge to be erased.
/// 
template<typename VISITOR>
inline void 
Graph<VISITOR>::eraseEdge(
    const unsigned long edgeIndex
) {
    assert(edgeIndex < numberOfEdges()); 

    eraseAdjacenciesForEdge(edgeIndex);
    if(edgeIndex == numberOfEdges() - 1) { // if the last edge is erased
        edges_.pop_back(); // delete
        visitor_.eraseEdge(edgeIndex);
    }
    else { 
        unsigned long movingEdgeIndex = numberOfEdges() - 1;
        eraseAdjacenciesForEdge(movingEdgeIndex);
        edges_[edgeIndex] = edges_[movingEdgeIndex]; // copy
        edges_.pop_back(); // delete
        insertAdjacenciesForEdge(edgeIndex);
        visitor_.eraseEdge(edgeIndex);
        visitor_.relabelEdge(movingEdgeIndex, edgeIndex);
    }
}

/// Get an iterator to the beginning of the sequence of vertices reachable from a given vertex via a single edge.
///
/// \param vertex Integer index of the vertex.
/// \return VertexIterator.
/// 
/// \sa verticesFromVertexEnd()
///
template<typename VISITOR>
inline typename Graph<VISITOR>::VertexIterator 
Graph<VISITOR>::verticesFromVertexBegin(
    const unsigned long vertex
) const { 
    return vertices_[vertex].begin(); 
}

/// Get an iterator to the end of the sequence of vertices reachable from a given vertex via a single edge.
///
/// \param vertex Integer index of the vertex.
/// \return VertexIterator.
/// 
/// \sa verticesFromVertexBegin()
/// 
template<typename VISITOR>
inline typename Graph<VISITOR>::VertexIterator 
Graph<VISITOR>::verticesFromVertexEnd(
    const unsigned long vertex
) const { 
    return vertices_[vertex].end(); 
}

/// Get an iterator to the beginning of the sequence of vertices from which a given vertex is reachable via a single edge.
///
/// \param vertex Integer index of the vertex.
/// \return VertexIterator.
/// 
/// \sa verticesToVertexEnd()
///
template<typename VISITOR>
inline typename Graph<VISITOR>::VertexIterator 
Graph<VISITOR>::verticesToVertexBegin(
    const unsigned long vertex
) const { 
    return vertices_[vertex].begin(); 
}

/// Get an iterator to the end of the sequence of vertices from which a given vertex is reachable via a single edge.
///
/// \param vertex Integer index of the vertex.
/// \return VertexIterator.
/// 
/// \sa verticesToVertexBegin()
///
template<typename VISITOR>
inline typename Graph<VISITOR>::VertexIterator 
Graph<VISITOR>::verticesToVertexEnd(
    const unsigned long vertex
) const { 
    return vertices_[vertex].end(); 
}

/// Get an iterator to the beginning of the sequence of edges that originate from a given vertex.
///
/// \param vertex Integer index of the vertex.
/// \return EdgeIterator.
///
/// \sa edgesFromVertexEnd()
///
template<typename VISITOR>
inline typename Graph<VISITOR>::EdgeIterator 
Graph<VISITOR>::edgesFromVertexBegin(
    const unsigned long vertex
) const { 
    return vertices_[vertex].begin(); 
}

/// Get an iterator to the end of the sequence of edges that originate from a given vertex.
///
/// \param vertex Integer index of the vertex.
/// \return EdgeIterator.
///
/// \sa edgesFromVertexBegin()
///
template<typename VISITOR>
inline typename Graph<VISITOR>::EdgeIterator 
Graph<VISITOR>::edgesFromVertexEnd(
    const unsigned long vertex
) const { 
    return vertices_[vertex].end(); 
}

/// Get an iterator to the beginning of the sequence of edges that are incident to a given vertex.
///
/// \param vertex Integer index of the vertex.
/// \return EdgeIterator.
///
/// \sa edgesToVertexEnd()
///
template<typename VISITOR>
inline typename Graph<VISITOR>::EdgeIterator 
Graph<VISITOR>::edgesToVertexBegin(
    const unsigned long vertex
) const { 
    return vertices_[vertex].begin(); 
}

/// Get an iterator to the end of the sequence of edges that are incident to a given vertex.
///
/// \param vertex Integer index of the vertex.
/// \return EdgeIterator.
///
/// \sa edgesToVertexBegin()
///
template<typename VISITOR>
inline typename Graph<VISITOR>::EdgeIterator 
Graph<VISITOR>::edgesToVertexEnd(
    const unsigned long vertex
) const { 
    return vertices_[vertex].end(); 
}

/// Get an iterator to the beginning of the sequence of adjacencies that originate from a given vertex.
///
/// \param vertex Integer index of the vertex.
/// \return AdjacencyIterator.
///
/// \sa adjacenciesFromVertexEnd()
///
template<typename VISITOR>
inline typename Graph<VISITOR>::AdjacencyIterator 
Graph<VISITOR>::adjacenciesFromVertexBegin(
    const unsigned long vertex
) const {
    return vertices_[vertex].begin();
}

/// Get an iterator to the end of the sequence of adjacencies that originate from a given vertex.
///
/// \param vertex Integer index of the vertex.
/// \return AdjacencyIterator.
///
/// \sa adjacenciesFromVertexBegin()
///
template<typename VISITOR>
inline typename Graph<VISITOR>::AdjacencyIterator 
Graph<VISITOR>::adjacenciesFromVertexEnd(
    const unsigned long vertex
) const {
    return vertices_[vertex].end();
}

/// Get an iterator to the beginning of the sequence of adjacencies incident to a given vertex.
///
/// \param vertex Integer index of the vertex.
/// \return AdjacencyIterator.
///
/// \sa adjacenciesToVertexEnd()
///
template<typename VISITOR>
inline typename Graph<VISITOR>::AdjacencyIterator 
Graph<VISITOR>::adjacenciesToVertexBegin(
    const unsigned long vertex
) const {
    return vertices_[vertex].begin();
}

/// Get an iterator to the end of the sequence of adjacencies incident to a given vertex.
///
/// \param vertex Integer index of the vertex.
/// \return AdjacencyIterator.
///
/// \sa adjacenciesToVertexBegin()
///
template<typename VISITOR>
inline typename Graph<VISITOR>::AdjacencyIterator 
Graph<VISITOR>::adjacenciesToVertexEnd(
    const unsigned long vertex
) const {
    return vertices_[vertex].end();
}

/// Reserve memory for at least the given total number of vertices.
///
/// \param number Total number of vertices.
///
template<typename VISITOR>
inline void 
Graph<VISITOR>::reserveVertices(
    const unsigned long number
) {
    vertices_.reserve(number);
}

/// Reserve memory for at least the given total number of edges.
///
/// \param number Total number of edges.
///
template<typename VISITOR>
inline void 
Graph<VISITOR>::reserveEdges(
    const unsigned long number
) {
    edges_.reserve(number);
}

/// Get the j-th adjacency from a vertex.
///
/// \param vertex Vertex.
/// \param j Number of the adjacency.
///
template<typename VISITOR>
inline const typename Graph<VISITOR>::AdjacencyType&
Graph<VISITOR>::adjacencyFromVertex(
    const unsigned long vertex,
    const unsigned long j
) const {
    return vertices_[vertex][j];
}

/// Get the j-th adjacency to a vertex.
///
/// \param vertex Vertex.
/// \param j Number of the adjacency.
///
template<typename VISITOR>
inline const typename Graph<VISITOR>::AdjacencyType&
Graph<VISITOR>::adjacencyToVertex(
    const unsigned long vertex,
    const unsigned long j
) const {
    return vertices_[vertex][j];
}

/// Search for an edge (in logarithmic time).
///
/// \param vertex0 first vertex of the edge.
/// \param vertex1 second vertex of the edge.
/// \return if an edge from vertex0 to vertex1 exists, pair.first is true 
///     and pair.second is the index of such an edge. if no edge from vertex0
///     to vertex1 exists, pair.first is false and pair.second is undefined.
///
template<typename VISITOR>
inline std::pair<bool, unsigned long>
Graph<VISITOR>::findEdge(
    const unsigned long vertex0,
    const unsigned long vertex1
) const {
    assert(vertex0 < numberOfVertices());
    assert(vertex1 < numberOfVertices());

    unsigned long v0 = vertex0;
    unsigned long v1 = vertex1;
    if(numberOfEdgesFromVertex(vertex1) < numberOfEdgesFromVertex(vertex0)) {
        v0 = vertex1;
        v1 = vertex0;
    }
    VertexIterator it = std::lower_bound(
        verticesFromVertexBegin(v0),
        verticesFromVertexEnd(v0),
        v1
    ); // binary search
    if(it != verticesFromVertexEnd(v0) && *it == v1) {
        // access the corresponding edge in constant time
        const unsigned long j = std::distance(verticesFromVertexBegin(v0), it);
        return std::make_pair(true, edgeFromVertex(v0, j));
    }
    else {
        return std::make_pair(false, 0);
    }
}
    
template<typename VISITOR>
inline void
Graph<VISITOR>::getVerticesOfEdge(const unsigned long edgeIndex, unsigned long &v1, unsigned long &v2)
{
    if(edgeIndex >= numberOfEdges())
    {
        printf("The edge doesn't exist!\n");
    }
    else
    {
        v1 = edges_[edgeIndex][0];
        v2 = edges_[edgeIndex][1];
    }
}

/// Indicate if multiple edges are enabled.
///
/// \return true if multiple edges are enabled, false otherwise.
///
template<typename VISITOR>
inline bool
Graph<VISITOR>::multipleEdgesEnabled() const {
    return multipleEdgesEnabled_;
}

/// Indicate if multiple edges are enabled.
///
/// Enable multiple edges like this: graph.multipleEdgesEnabled() = true;
///
/// \return reference the a Boolean flag.
///
template<typename VISITOR>
inline bool&
Graph<VISITOR>::multipleEdgesEnabled() {
    return multipleEdgesEnabled_;
}

template<typename VISITOR>
inline void 
Graph<VISITOR>::insertAdjacenciesForEdge(
    const unsigned long edgeIndex
) {
    const Edge& edge = edges_[edgeIndex];
    const unsigned long vertexIndex0 = edge[0];
    const unsigned long vertexIndex1 = edge[1];
    vertices_[vertexIndex0].insert(
        AdjacencyType(vertexIndex1, edgeIndex)
    );
    if(vertexIndex1 != vertexIndex0) {
        vertices_[vertexIndex1].insert(
            AdjacencyType(vertexIndex0, edgeIndex)
        );
    }
}

template<typename VISITOR>
inline void 
Graph<VISITOR>::eraseAdjacenciesForEdge(
    const unsigned long edgeIndex
) {
    const Edge& edge = edges_[edgeIndex];
    const unsigned long vertexIndex0 = edge[0];
    const unsigned long vertexIndex1 = edge[1];
    Vertex& vertex0 = vertices_[vertexIndex0];
    Vertex& vertex1 = vertices_[vertexIndex1];

    AdjacencyType adj(vertexIndex1, edgeIndex);
    RandomAccessSet<AdjacencyType>::iterator it = vertex0.find(adj);
    assert(it != vertex0.end()); 
    vertex0.erase(it);
    
    if(vertexIndex1 != vertexIndex0) { // if not a self-edge
        adj.vertex() = vertexIndex0;
        it = vertex1.find(adj);
        assert(it != vertex1.end()); 
        vertex1.erase(it);
    }
}

} // namespace graph
} // namespace andres

#endif // #ifndef ANDRES_GRAPH_HXX
