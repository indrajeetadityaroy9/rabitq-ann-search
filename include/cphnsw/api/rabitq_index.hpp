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
#include "../graph/visitation_table.hpp"
#include "params.hpp"
#include <vector>
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
    static constexpr size_t DIMS = D;
    static constexpr size_t DEGREE = R;
    static constexpr size_t BIT_WIDTH = BitWidth;

    explicit RaBitQIndex(const IndexParams& params)
        : params_(params)
        , encoder_(params.dim, params.seed)
        , graph_(params.dim) {
        if (params.dim == 0) throw std::invalid_argument("dim must be > 0");
    }

    RaBitQIndex(const IndexParams& params, Graph&& graph)
        : params_(params)
        , encoder_(params.dim, params.seed)
        , graph_(std::move(graph))
        , finalized_(true) {
        if (params.dim == 0) throw std::invalid_argument("dim must be > 0");
    }

    void add_batch(const float* vecs, size_t num_vecs) {
        if (num_vecs == 0) return;

        graph_.reserve(graph_.size() + num_vecs);

        std::vector<CodeType> codes(num_vecs);
        encoder_.encode_batch(vecs, num_vecs, codes.data());

        for (size_t i = 0; i < num_vecs; ++i) {
            graph_.add_node(codes[i], vecs + i * params_.dim);
        }

        needs_build_ = true;
    }

    void finalize(const BuildParams& params) {
        if (needs_build_) {
            graph_refinement::optimize_graph_adaptive(
                graph_, encoder_, params.num_threads, params.verbose, params_.seed);
            needs_build_ = false;
        }
        finalized_ = true;
    }

    std::vector<SearchResult> search(
        const float* query,
        const SearchParams& params = SearchParams()) const
    {
        if (graph_.empty()) return {};

        float gamma = (params.gamma_override >= 0.0f)
            ? params.gamma_override
            : adaptive_defaults::gamma_from_recall(params.recall_target, D);
        float eps = adaptive_defaults::error_epsilon_search(params.recall_target);

        QueryType encoded = encoder_.encode_query_raw(query);
        encoded.error_epsilon = eps;

        thread_local TwoLevelVisitationTable visited(0);
        if (visited.capacity() < graph_.size()) {
            visited.resize(graph_.size() + adaptive_defaults::visitation_headroom(graph_.size()));
        }
        return rabitq_search::search<D, R, BitWidth>(encoded, query, graph_, params.k, gamma, visited);
    }

    size_t size() const { return graph_.size(); }
    size_t dim() const { return params_.dim; }
    bool is_finalized() const { return finalized_; }
    const Graph& graph() const { return graph_; }

private:
    IndexParams params_;
    Encoder encoder_;
    Graph graph_;
    bool finalized_ = false;
    bool needs_build_ = false;
};

}  // namespace cphnsw
