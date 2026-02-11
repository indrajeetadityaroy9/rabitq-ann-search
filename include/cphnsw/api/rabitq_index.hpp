#pragma once

#include "../core/types.hpp"
#include "../core/codes.hpp"
#include "../core/memory.hpp"
#include "../core/adaptive_defaults.hpp"
#include "../encoder/rabitq_encoder.hpp"
#include "../distance/fastscan_kernel.hpp"
#include "../graph/rabitq_graph.hpp"
#include "../graph/graph_refinement.hpp"
#include "../search/rabitq_search.hpp"
#include "params.hpp"
#include <vector>
#include <random>
#include <stdexcept>
#include <type_traits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cphnsw {

template <size_t D, size_t R = 32, size_t BitWidth = 1, typename RotationPolicy = RandomHadamardRotation>
class RaBitQIndex {
public:
    using CodeType = std::conditional_t<BitWidth == 1,
        RaBitQCode<D>, NbitRaBitQCode<D, BitWidth>>;
    using QueryType = RaBitQQuery<D>;
    using Encoder = std::conditional_t<BitWidth == 1,
        RaBitQEncoder<D, RotationPolicy>,
        NbitRaBitQEncoder<D, BitWidth, RotationPolicy>>;
    using Graph = RaBitQGraph<D, R, BitWidth>;
    using Engine = RaBitQSearchEngine<D, R, BitWidth>;
    using Refinement = GraphRefinement<D, R, BitWidth>;

    static constexpr size_t DIMS = D;
    static constexpr size_t DEGREE = R;
    static constexpr size_t BIT_WIDTH = BitWidth;

    explicit RaBitQIndex(const IndexParams& params)
        : params_(params)
        , encoder_(params.dim, params.seed)
        , graph_(params.dim) {
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

        needs_build_ = true;
        return id;
    }

    void add_batch(const float* vecs, size_t num_vecs,
                   const BuildParams& build_params = BuildParams()) {
        if (num_vecs == 0) return;

        graph_.reserve(graph_.size() + num_vecs);

        std::vector<CodeType> codes(num_vecs);
        encoder_.encode_batch(vecs, num_vecs, codes.data());

        for (size_t i = 0; i < num_vecs; ++i) {
            graph_.add_node(codes[i], vecs + i * params_.dim);
        }

        needs_build_ = true;

        if (build_params.verbose) {
            printf("[RaBitQIndex] Added %zu vectors (B=%zu). Edges will be built in finalize().\n",
                   num_vecs, BitWidth);
        }
    }

    void finalize(const BuildParams& params) {
        if (needs_build_) {
            // Resolve adaptive defaults from sentinels
            size_t ef_c = params.ef_construction > 0
                ? params.ef_construction
                : AdaptiveDefaults::ef_construction(graph_.size(), R);
            float err_tol = params.error_tolerance >= 0.0f
                ? params.error_tolerance
                : AdaptiveDefaults::error_tolerance(D);
            float err_eps = params.error_epsilon > 0.0f
                ? params.error_epsilon
                : AdaptiveDefaults::ERROR_EPSILON_BUILD;

            if (params.verbose)
                printf("[RaBitQIndex] Running graph optimization (ef_c=%zu, err_tol=%.4f, err_eps=%.3f)...\n",
                       ef_c, err_tol, err_eps);

            Refinement::optimize_graph_adaptive(graph_, encoder_,
                ef_c, err_tol, err_eps, params.num_threads, params.verbose);

            needs_build_ = false;
        }
        finalized_ = true;
    }

    void finalize() {
        finalize(BuildParams{});
    }

    std::vector<SearchResult> search(
        const float* query,
        const SearchParams& params = SearchParams()) const
    {
        if (graph_.empty()) return {};

        // Resolve adaptive defaults from sentinels
        size_t ef = params.ef > 0
            ? params.ef
            : AdaptiveDefaults::ef_search(params.k, params.recall_target);
        float eps = params.error_epsilon > 0.0f
            ? params.error_epsilon
            : AdaptiveDefaults::error_epsilon_search(params.recall_target);

        QueryType encoded = encoder_.encode_query(query);
        encoded.error_epsilon = eps;

        return Engine::search(encoded, query, graph_, ef, params.k);
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

}  // namespace cphnsw
