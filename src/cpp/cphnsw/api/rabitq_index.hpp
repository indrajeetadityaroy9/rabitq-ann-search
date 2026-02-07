#pragma once

#include "../core/types.hpp"
#include "../core/codes.hpp"
#include "../core/memory.hpp"
#include "../encoder/rabitq_encoder.hpp"
#include "../distance/fastscan_kernel.hpp"
#include "../graph/rabitq_graph.hpp"
#include "../graph/graph_refinement.hpp"
#include "../search/rabitq_search.hpp"
#include "params.hpp"
#include <vector>
#include <random>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cphnsw {

// ============================================================================
// RaBitQIndex: Unified Index with RaBitQ Quantization
// ============================================================================

/**
 * RaBitQIndex: High-performance ANN index combining:
 *   - RaBitQ (SIGMOD 2024) for theoretically-grounded binary quantization
 *   - SymphonyQG (SIGMOD 2025) for FastScan SIMD and implicit reranking
 *   - HNSW-style greedy beam search for graph traversal
 *
 * USAGE:
 *   RaBitQIndex<128> index(IndexParams().set_dim(100));
 *   index.add_batch(vectors, num_vectors);
 *   index.finalize();  // Graph refinement + medoid selection
 *   auto results = index.search(query, SearchParams().set_k(10));
 *
 * DISTANCE PROPERTIES:
 *   - Unbiased L2 distance estimator (E[est] = true_dist)
 *   - O(1/sqrt(D)) error bound with high probability
 *   - Implicit reranking: exact distances computed at visit time
 *
 * @tparam D Padded dimension (next power of 2 of input dim)
 * @tparam R Fixed graph degree (multiple of 32)
 */
template <size_t D, size_t R = 32>
class RaBitQIndex {
public:
    using CodeType = RaBitQCode<D>;
    using QueryType = RaBitQQuery<D>;
    using Encoder = RaBitQEncoder<D>;
    using Graph = RaBitQGraph<D, R>;
    using Engine = RaBitQSearchEngine<D, R>;
    using Refinement = GraphRefinement<D, R>;

    static constexpr size_t DIMS = D;
    static constexpr size_t DEGREE = R;

    /**
     * Construct index with parameters.
     */
    explicit RaBitQIndex(const IndexParams& params)
        : params_(params)
        , encoder_(params.dim, params.seed)
        , graph_(params.dim, params.initial_capacity) {
        if (params.dim == 0) throw std::invalid_argument("dim must be > 0");
    }

    /**
     * Construct index with just dimension.
     */
    explicit RaBitQIndex(size_t dim)
        : RaBitQIndex(IndexParams().set_dim(dim)) {}

    // ========================================================================
    // Index Construction
    // ========================================================================

    /**
     * Add a single vector to the index.
     */
    NodeId add(const float* vec) {
        // Ensure centroid is set (use zero centroid for single adds)
        if (!encoder_.has_centroid()) {
            std::vector<float> zero(params_.dim, 0.0f);
            encoder_.set_centroid(zero.data());
        }

        CodeType code = encoder_.encode(vec);
        NodeId id = graph_.add_node(code, vec);

        // Build edges for non-first nodes
        if (id > 0) {
            insert_into_graph(id, vec, code);
        }

        return id;
    }

    /**
     * Add a batch of vectors (recommended for best performance).
     *
     * Computes centroid from the batch, encodes all vectors, then
     * builds graph edges in parallel.
     */
    void add_batch(const float* vecs, size_t num_vecs,
                   const BuildParams& build_params = BuildParams()) {
        if (num_vecs == 0) return;

        // Reserve capacity
        graph_.reserve(graph_.size() + num_vecs);

        // Compute centroid and encode
        std::vector<CodeType> codes(num_vecs);
        encoder_.encode_batch(vecs, num_vecs, codes.data());

        // Add nodes sequentially (thread-safe but sequential)
        std::vector<NodeId> ids(num_vecs);
        for (size_t i = 0; i < num_vecs; ++i) {
            ids[i] = graph_.add_node(codes[i], vecs + i * params_.dim);
        }

        // Build edges in parallel
        size_t num_threads = build_params.num_threads;
#ifdef _OPENMP
        if (num_threads == 0) num_threads = omp_get_max_threads();

        #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
        for (size_t i = 0; i < num_vecs; ++i) {
            if (ids[i] > 0) {
                insert_into_graph(ids[i], vecs + i * params_.dim, codes[i]);
            }
        }
#else
        for (size_t i = 0; i < num_vecs; ++i) {
            if (ids[i] > 0) {
                insert_into_graph(ids[i], vecs + i * params_.dim, codes[i]);
            }
        }
#endif

        if (build_params.verbose) {
            printf("[RaBitQIndex] Added %zu vectors, total=%zu, avg_degree=%.1f\n",
                   num_vecs, graph_.size(), graph_.average_degree());
        }
    }

    /**
     * Finalize the index: runs graph refinement and medoid selection.
     * Call this after all vectors have been added.
     */
    void finalize(bool verbose = false) {
        Refinement::refine(graph_, 0, verbose);
        finalized_ = true;
    }

    // ========================================================================
    // Search
    // ========================================================================

    /**
     * Search for k-nearest neighbors.
     *
     * @param query Raw query vector
     * @param params Search parameters (k, ef, etc.)
     * @return k-nearest neighbors sorted by distance
     */
    std::vector<SearchResult> search(
        const float* query,
        const SearchParams& params = SearchParams()) const
    {
        if (graph_.empty()) return {};

        // Encode query
        QueryType encoded = encoder_.encode_query(query);

        // Search with implicit reranking
        return Engine::search(
            encoded, query, graph_,
            params.ef, params.k);
    }

    /**
     * Search with just k parameter.
     */
    std::vector<SearchResult> search(const float* query, size_t k) const {
        return search(query, SearchParams().set_k(k));
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    size_t size() const { return graph_.size(); }
    bool empty() const { return graph_.empty(); }
    size_t dim() const { return params_.dim; }
    bool is_finalized() const { return finalized_; }
    const IndexParams& params() const { return params_; }
    const Graph& graph() const { return graph_; }
    const Encoder& encoder() const { return encoder_; }

    struct Stats {
        size_t num_nodes;
        float avg_degree;
        size_t max_degree;
        size_t isolated_nodes;
    };

    Stats get_stats() const {
        return Stats{
            graph_.size(),
            graph_.average_degree(),
            graph_.max_degree(),
            graph_.count_isolated()
        };
    }

private:
    IndexParams params_;
    Encoder encoder_;
    Graph graph_;
    bool finalized_ = false;

    /**
     * Insert a node into the graph by searching for neighbors
     * and creating bidirectional edges.
     */
    void insert_into_graph(NodeId id, const float* vec, const CodeType& code) {
        // Encode query for searching
        QueryType query = encoder_.encode_query(vec);

        // Greedy search from entry point to find candidate neighbors.
        // Uses RaBitQ estimated distances during construction.
        size_t n = graph_.size();
        size_t ef = params_.ef_construction;
        size_t M = params_.M;
        if (M > R) M = R;

        // Find M closest nodes
        std::vector<std::pair<float, NodeId>> candidates;
        candidates.reserve(std::min(n, ef));

        // Use entry point + greedy expansion
        NodeId ep = graph_.entry_point();
        if (ep == INVALID_NODE || ep == id) return;

        // Simple greedy search from entry point
        thread_local VisitationTable visited(0);
        if (visited.capacity() < n + 1024) {
            visited.resize(n + 1024);
        }
        uint64_t qid = visited.new_query();

        // BFS-like expansion from entry point
        std::priority_queue<std::pair<float, NodeId>,
                           std::vector<std::pair<float, NodeId>>,
                           std::greater<>> pq;

        float ep_dist = RaBitQMetricPolicy<D>::compute_distance(query, graph_.get_code(ep));
        pq.push({ep_dist, ep});
        visited.check_and_mark(ep, qid);

        std::vector<std::pair<float, NodeId>> found;
        found.push_back({ep_dist, ep});

        while (!pq.empty() && found.size() < ef) {
            auto [dist, node] = pq.top();
            pq.pop();

            const auto& nb = graph_.get_neighbors(node);
            for (size_t i = 0; i < nb.size(); ++i) {
                NodeId nid = nb.neighbor_ids[i];
                if (nid == INVALID_NODE || nid == id) continue;
                if (visited.check_and_mark(nid, qid)) continue;

                float d = RaBitQMetricPolicy<D>::compute_distance(query, graph_.get_code(nid));
                pq.push({d, nid});
                found.push_back({d, nid});
            }
        }

        // Sort by distance and take M closest
        std::sort(found.begin(), found.end());
        if (found.size() > M) found.resize(M);

        // Add bidirectional edges
        for (size_t i = 0; i < found.size(); ++i) {
            NodeId neighbor_id = found[i].second;

            // Edge: id → neighbor
            VertexAuxData aux_fwd;
            aux_fwd.dist_to_centroid = graph_.get_code(neighbor_id).dist_to_centroid;
            aux_fwd.ip_quantized_original = graph_.get_code(neighbor_id).ip_quantized_original;
            aux_fwd.ip_xbar_Pinv_c = 0.0f;  // Simplified for construction

            auto& nb_fwd = graph_.get_neighbors(id);
            if (nb_fwd.size() < R) {
                graph_.set_neighbor(id, nb_fwd.size(), neighbor_id,
                                   graph_.get_code(neighbor_id).signs, aux_fwd);
            }

            // Edge: neighbor → id
            VertexAuxData aux_rev;
            aux_rev.dist_to_centroid = code.dist_to_centroid;
            aux_rev.ip_quantized_original = code.ip_quantized_original;
            aux_rev.ip_xbar_Pinv_c = 0.0f;

            auto& nb_rev = graph_.get_neighbors(neighbor_id);
            if (nb_rev.size() < R) {
                graph_.set_neighbor(neighbor_id, nb_rev.size(), id,
                                   code.signs, aux_rev);
            }
        }
    }
};

// ============================================================================
// Type Aliases for Common Configurations
// ============================================================================

using RaBitQIndex128 = RaBitQIndex<128, 32>;   // SIFT, GloVe-100
using RaBitQIndex256 = RaBitQIndex<256, 32>;   // GloVe-200
using RaBitQIndex512 = RaBitQIndex<512, 32>;   // Mid-range
using RaBitQIndex1024 = RaBitQIndex<1024, 32>; // Text embeddings

}  // namespace cphnsw
