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

template <size_t D, size_t R = 32, size_t BitWidth = 1>
class RaBitQIndex {
public:
    using CodeType = std::conditional_t<BitWidth == 1,
        RaBitQCode<D>, NbitRaBitQCode<D, BitWidth>>;
    using QueryType = RaBitQQuery<D>;
    using Encoder = std::conditional_t<BitWidth == 1,
        RaBitQEncoder<D>,
        NbitRaBitQEncoder<D, BitWidth>>;
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

        (void)build_params;
    }

    void finalize(const BuildParams& params) {
        if (needs_build_) {
            Refinement::optimize_graph_adaptive(
                graph_, encoder_, params.num_threads, params.verbose);
            needs_build_ = false;
        }
        finalized_ = true;
    }

    std::vector<SearchResult> search(
        const float* query,
        const SearchParams& params = SearchParams()) const
    {
        if (graph_.empty()) return {};

        float gamma = AdaptiveDefaults::gamma_from_recall(params.recall_target, D);
        float eps = AdaptiveDefaults::error_epsilon_search(params.recall_target);

        // SymphonyQG: use raw query LUT (no centering/normalizing)
        // for parent-relative edge distance estimation
        QueryType encoded = encoder_.encode_query_raw(query);
        encoded.error_epsilon = eps;

        return Engine::search(encoded, query, graph_, params.k, gamma);
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
