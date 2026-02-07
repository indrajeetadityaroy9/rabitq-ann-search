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

template <size_t D, size_t R = 32, typename RotationPolicy = RandomHadamardRotation>
class RaBitQIndex {
public:
    using CodeType = RaBitQCode<D>;
    using QueryType = RaBitQQuery<D>;
    using Encoder = RaBitQEncoder<D, RotationPolicy>;
    using Graph = RaBitQGraph<D, R>;
    using Engine = RaBitQSearchEngine<D, R>;
    using Refinement = GraphRefinement<D, R>;

    static constexpr size_t DIMS = D;
    static constexpr size_t DEGREE = R;

    explicit RaBitQIndex(const IndexParams& params)
        : params_(params)
        , encoder_(params.dim, params.seed)
        , graph_(params.dim, params.initial_capacity) {
        if (params.dim == 0) throw std::invalid_argument("dim must be > 0");
    }

    explicit RaBitQIndex(size_t dim)
        : RaBitQIndex(IndexParams().set_dim(dim)) {}

    NodeId add(const float* vec) {
        if (!encoder_.has_centroid()) {
            std::vector<float> zero(params_.dim, 0.0f);
            encoder_.set_centroid(zero.data());
        }

        CodeType code = encoder_.encode(vec);
        NodeId id = graph_.add_node(code, vec);
        
        // Mark as needing optimization. 
        // Edges will be built in finalize().
        needs_build_ = true;

        return id;
    }

    void add_batch(const float* vecs, size_t num_vecs,
                   const BuildParams& build_params = BuildParams()) {
        if (num_vecs == 0) return;

        encoder_.set_stochastic_rounding(build_params.stochastic_rounding);

        graph_.reserve(graph_.size() + num_vecs);

        std::vector<CodeType> codes(num_vecs);
        encoder_.encode_batch(vecs, num_vecs, codes.data());

        for (size_t i = 0; i < num_vecs; ++i) {
            graph_.add_node(codes[i], vecs + i * params_.dim);
        }

        needs_build_ = true;

        if (build_params.verbose) {
            printf("[RaBitQIndex] Added %zu vectors. Edges will be built in finalize().\n", num_vecs);
        }
    }

    void finalize(size_t num_threads = 0, bool verbose = false) {
        if (needs_build_) {
            if (verbose) printf("[RaBitQIndex] Running graph optimization...\n");
            Refinement::optimize_graph(graph_, encoder_, params_.ef_construction, num_threads, verbose);
            needs_build_ = false;
        }
        finalized_ = true;
    }

    std::vector<SearchResult> search(
        const float* query,
        const SearchParams& params = SearchParams()) const
    {
        if (graph_.empty()) return {};

        QueryType encoded = encoder_.encode_query(query);

        return Engine::search(
            encoded, query, graph_,
            params.ef, params.k);
    }

    std::vector<SearchResult> search(const float* query, size_t k) const {
        return search(query, SearchParams().set_k(k));
    }

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
    bool needs_build_ = false;


};

using RaBitQIndex128 = RaBitQIndex<128, 32>;
using RaBitQIndex256 = RaBitQIndex<256, 32>;
using RaBitQIndex512 = RaBitQIndex<512, 32>;
using RaBitQIndex1024 = RaBitQIndex<1024, 32>;

// Dense (SOTA) variants
using RaBitQIndexDense128 = RaBitQIndex<128, 32, DenseRotation>;
using RaBitQIndexDense256 = RaBitQIndex<256, 32, DenseRotation>;
using RaBitQIndexDense512 = RaBitQIndex<512, 32, DenseRotation>;
using RaBitQIndexDense1024 = RaBitQIndex<1024, 32, DenseRotation>;

}  // namespace cphnsw