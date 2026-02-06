#pragma once

#include "../core/codes.hpp"
#include "../core/memory.hpp"
#include "../distance/metric_policy.hpp"
#include "../graph/flat_graph.hpp"
#include "../search/search_engine.hpp"
#include "../encoder/cp_encoder.hpp"
#include <vector>
#include <random>
#include <stdexcept>
#include <omp.h>

namespace cphnsw {

// ============================================================================
// Index Configuration
// ============================================================================

struct IndexParams {
    size_t dim = 0;
    size_t M = 32;
    size_t ef_construction = 200;
    uint64_t seed = 42;
    size_t initial_capacity = 1024;

    IndexParams& set_dim(size_t d) { dim = d; return *this; }
    IndexParams& set_M(size_t m) { M = m; return *this; }
    IndexParams& set_ef_construction(size_t ef) { ef_construction = ef; return *this; }
    IndexParams& set_seed(uint64_t s) { seed = s; return *this; }
    IndexParams& set_capacity(size_t c) { initial_capacity = c; return *this; }
};

struct BuildParams {
    size_t num_threads = 0;
    bool verbose = false;

    BuildParams& set_threads(size_t n) { num_threads = n; return *this; }
    BuildParams& set_verbose(bool v) { verbose = v; return *this; }
};

struct SearchParams {
    size_t k = 10;
    size_t ef = 100;
    size_t num_entry_points = 1;
    bool rerank = true;
    size_t rerank_k = 200;

    SearchParams& set_k(size_t num) { k = num; return *this; }
    SearchParams& set_ef(size_t e) { ef = e; return *this; }
    SearchParams& set_rerank(bool r, size_t rk = 200) { rerank = r; rerank_k = rk; return *this; }
};

// ============================================================================
// CPHNSWIndex: Unified Index API
// ============================================================================

template <size_t K, size_t R = 0, int Shift = 2>
class CPHNSWIndex {
public:
    using CodeType = ResidualCode<K, R>;
    using QueryType = CodeQuery<K, R, Shift>;
    using Policy = UnifiedMetricPolicy<K, R, Shift>;
    using Graph = FlatGraph<CodeType>;
    using Encoder = CPEncoder<K, R>;
    using Engine = SearchEngine<Policy>;

    static constexpr size_t PRIMARY_BITS = K;
    static constexpr size_t RESIDUAL_BITS = R;
    static constexpr int WEIGHT_SHIFT = Shift;
    static constexpr bool HAS_RESIDUAL = (R > 0);

    explicit CPHNSWIndex(const IndexParams& params)
        : params_(params)
        , encoder_(params.dim, params.seed)
        , graph_(params.initial_capacity, params.M)
        , rng_(params.seed) {
        if (params.dim == 0) throw std::invalid_argument("dim must be > 0");
    }

    explicit CPHNSWIndex(size_t dim)
        : CPHNSWIndex(IndexParams().set_dim(dim)) {}

    NodeId add(const float* vec) {
        CodeType code = encoder_.encode(vec);
        NodeId id = graph_.add_node(code);
        if (id > 0) insert_into_graph(id, vec, code);
        {
            std::lock_guard<std::mutex> lock(vectors_mutex_);
            if (id >= vectors_.size()) vectors_.resize(id + 1);
            vectors_[id].assign(vec, vec + params_.dim);
        }
        return id;
    }

    void add_batch(const float* vecs, size_t num_vecs,
                   const BuildParams& build_params = BuildParams()) {
        if (num_vecs == 0) return;
        graph_.reserve(graph_.size() + num_vecs);
        vectors_.resize(graph_.size() + num_vecs);
        std::vector<CodeType> codes(num_vecs);
        encoder_.encode_batch(vecs, num_vecs, codes.data());
        std::vector<NodeId> ids(num_vecs);
        for (size_t i = 0; i < num_vecs; ++i) {
            ids[i] = graph_.add_node(codes[i]);
            vectors_[ids[i]].assign(vecs + i * params_.dim, vecs + (i + 1) * params_.dim);
        }
        size_t num_threads = build_params.num_threads > 0 ? build_params.num_threads : omp_get_max_threads();
        #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
        for (size_t i = 0; i < num_vecs; ++i) {
            if (ids[i] > 0) insert_into_graph(ids[i], vecs + i * params_.dim, codes[i]);
        }
    }

    std::vector<SearchResult> search(const float* query,
                                      const SearchParams& params = SearchParams()) const {
        if (graph_.empty()) return {};
        float avg_norm = compute_average_norm();
        QueryType encoded_query = encoder_.encode_query(query, avg_norm);
        std::vector<NodeId> entry_points;
        if (params.num_entry_points == 1) {
            entry_points.push_back(graph_.entry_point());
        } else {
            std::mt19937_64 local_rng(rng_());
            entry_points = graph_.get_random_entry_points(params.num_entry_points, local_rng);
        }
        auto candidates = Engine::search(encoded_query, entry_points, params.ef, graph_, params.rerank ? params.rerank_k : params.k);
        if (params.rerank && !candidates.empty()) candidates = rerank(query, candidates, params.k);
        if (candidates.size() > params.k) candidates.resize(params.k);
        return candidates;
    }

    std::vector<SearchResult> search(const float* query, size_t k) const {
        return search(query, SearchParams().set_k(k));
    }

    size_t size() const { return graph_.size(); }
    bool empty() const { return graph_.empty(); }
    size_t dim() const { return params_.dim; }
    const IndexParams& params() const { return params_; }
    const Graph& graph() const { return graph_; }

    struct Stats {
        size_t num_nodes;
        float avg_degree;
        size_t max_degree;
        size_t isolated_nodes;
    };

    Stats get_stats() const {
        return Stats{ graph_.size(), graph_.average_degree(), graph_.max_degree(), graph_.count_isolated() };
    }

private:
    IndexParams params_;
    Encoder encoder_;
    Graph graph_;
    mutable std::mt19937_64 rng_;
    std::vector<std::vector<float>> vectors_;
    mutable std::mutex vectors_mutex_;
    mutable float cached_avg_norm_ = 1.0f;
    mutable bool norm_valid_ = false;

    void insert_into_graph(NodeId id, const float* vec, const CodeType& code) {
        QueryType query = encoder_.encode_query(vec, 1.0f);
        auto neighbors = Engine::search(query, graph_.entry_point(), params_.ef_construction, graph_, params_.M);
        for (const auto& neighbor : neighbors) {
            if (neighbor.id == id) continue;
            graph_.add_neighbor_safe(id, neighbor.id, graph_.get_code(neighbor.id), neighbor.distance);
            graph_.add_neighbor_safe(neighbor.id, id, code, neighbor.distance);
        }
    }

    std::vector<SearchResult> rerank(const float* query, const std::vector<SearchResult>& candidates, size_t k) const {
        std::vector<SearchResult> reranked;
        reranked.reserve(candidates.size());
        for (const auto& cand : candidates) {
            if (cand.id >= vectors_.size() || vectors_[cand.id].empty()) {
                reranked.push_back(cand);
                continue;
            }
            float dist = 0.0f;
            const auto& vec = vectors_[cand.id];
            for (size_t i = 0; i < params_.dim; ++i) {
                float diff = query[i] - vec[i];
                dist += diff * diff;
            }
            reranked.push_back({cand.id, dist});
        }
        std::sort(reranked.begin(), reranked.end());
        if (reranked.size() > k) reranked.resize(k);
        return reranked;
    }

    float compute_average_norm() const {
        if (norm_valid_) return cached_avg_norm_;
        if (vectors_.empty()) return 1.0f;
        double sum = 0.0;
        size_t count = 0;
        for (const auto& vec : vectors_) {
            if (vec.empty()) continue;
            double norm = 0.0;
            for (float x : vec) norm += x * x;
            sum += std::sqrt(norm);
            ++count;
        }
        if (count > 0) {
            cached_avg_norm_ = static_cast<float>(sum / count);
            norm_valid_ = true;
        }
        return cached_avg_norm_;
    }
};

using Index32 = CPHNSWIndex<32, 0>;
using Index64 = CPHNSWIndex<64, 0>;
using Index64_32 = CPHNSWIndex<64, 32, 2>;
using Index32_16 = CPHNSWIndex<32, 16, 2>;

}  // namespace cphnsw
