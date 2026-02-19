#pragma once

#include "../core/types.hpp"
#include "../core/codes.hpp"
#include "../core/memory.hpp"
#include "../core/adaptive_defaults.hpp"
#include "../core/evt_crc.hpp"
#include "../distance/fastscan_kernel.hpp"
#include "../encoder/rabitq_encoder.hpp"
#include "../graph/rabitq_graph.hpp"
#include "../graph/graph_refinement.hpp"
#include "../graph/neighbor_selection.hpp"
#include "../search/rabitq_search.hpp"
#include "../graph/visitation_table.hpp"
#include <vector>
#include <random>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <mutex>
#include <shared_mutex>
#include <limits>
#include <cstring>
#include <fstream>
#include <string>

#include <omp.h>

namespace cphnsw {

// All fields are populated by calibrate_estimator().
struct CalibrationSnapshot {
    // Affine correction (from Huber regression)
    float affine_a;
    float affine_b;
    float ip_qo_floor;

    // Distance distribution statistics
    float median_nn_dist_sq;
    float min_slack_sq;
    float median_nop;

    // EVT model
    EVTState evt;

    // Gamma bounds
    float gamma_min;
    float gamma_max;
    float gamma_beta;
    size_t gamma_warmup;
    int slack_levels;

    // Precomputed search parameters (max-recall, delta=1e-4)
    float search_ip_slack_levels[constants::kMaxSlackArray];
    int search_num_slack_levels;
    float search_gamma;
};


struct HNSWLayerEdge {
    NodeId node;
    std::vector<NodeId> neighbors;
    bool operator<(const HNSWLayerEdge& other) const { return node < other.node; }
};

template <size_t D, size_t R = 32, size_t BitWidth = 1>
class Index {
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

    static constexpr size_t M_UPPER = adaptive_defaults::upper_layer_degree(R, D);

    explicit Index(size_t dim)
        : dim_(dim)
        , encoder_(dim, constants::kDefaultRotationSeed)
        , graph_(dim)
        , mL_(1.0 / std::log(static_cast<double>(M_UPPER)))
        , rng_(constants::kDefaultLayerSeed)
    {
        if (dim == 0) throw std::invalid_argument("dim must be > 0");
    }

    void build(const float* vecs, size_t num_vecs) {
        std::unique_lock<std::shared_mutex> lock(index_mutex_);
        if (num_vecs == 0) {
            throw std::invalid_argument("build requires at least one vector.");
        }

        graph_ = Graph(dim_);
        calibration_ = {};
        finalized_ = false;
        needs_build_ = false;
        max_level_ = 0;
        entry_point_ = INVALID_NODE;
        upper_tau_ = 0.0f;
        node_levels_.clear();
        upper_layers_.clear();

        graph_.reserve(num_vecs);

        std::vector<CodeType> codes(num_vecs);
        encoder_.encode_batch(vecs, num_vecs, codes.data());

        for (size_t i = 0; i < num_vecs; ++i) {
            graph_.add_node(codes[i], vecs + i * dim_);
        }

        needs_build_ = true;
        finalized_ = false;
    }

    void finalize() {
        std::unique_lock<std::shared_mutex> lock(index_mutex_);
        size_t n = graph_.size();
        if (n == 0) {
            throw std::runtime_error("Cannot finalize an empty index.");
        }
        if (!needs_build_) {
            throw std::runtime_error("Finalize called without a pending build.");
        }

        // Derive metadata-based parameters
        profile_.derive(n, D, R, BitWidth);

        assign_layers(n);

        build_upper_layers();

        auto result = graph_refinement::optimize_graph_adaptive(
            graph_, encoder_);
        auto& perm = result.perm;
        profile_.graph_stats = result.stats;


        for (auto& layer : upper_layers_) {
            for (auto& edge : layer) {
                edge.node = perm.old_to_new[edge.node];
                for (auto& nb : edge.neighbors) {
                    nb = perm.old_to_new[nb];
                }
            }
            std::sort(layer.begin(), layer.end());
        }
        entry_point_ = perm.old_to_new[entry_point_];
        std::vector<int> new_levels(node_levels_.size());
        for (size_t i = 0; i < node_levels_.size(); ++i) {
            new_levels[perm.old_to_new[i]] = node_levels_[i];
        }
        node_levels_ = std::move(new_levels);

        size_t n_calib = std::min(profile_.min_calib_samples, n);
        calibrate_estimator(n_calib);

        needs_build_ = false;
        finalized_ = true;
    }

    std::vector<SearchResult> search(
        const float* query,
        size_t k = constants::kDefaultK) const
    {
        std::shared_lock<std::shared_mutex> lock(index_mutex_);

        thread_local AlignedVector<float> query_padded;
        query_padded.resize(D);
        std::memcpy(query_padded.data(), query, dim_ * sizeof(float));
        if (dim_ < D) {
            std::memset(query_padded.data() + dim_, 0, (D - dim_) * sizeof(float));
        }
        const float* query_vec = query_padded.data();

        QueryType encoded = encoder_.encode_query_raw(query_vec);
        encoded.affine_a = calibration_.affine_a;
        encoded.affine_b = calibration_.affine_b;
        encoded.ip_qo_floor = calibration_.ip_qo_floor;
        encoded.dot_slack = calibration_.search_ip_slack_levels[0];
        k = std::max<size_t>(k, 1);

        float gamma = calibration_.search_gamma;

        thread_local TwoLevelVisitationTable visited(0);
        if (visited.capacity() < graph_.size()) {
            visited.resize(graph_.size() + adaptive_defaults::visitation_headroom(graph_.size()));
        }

        NodeId ep = graph_.entry_point();
        if (max_level_ > 0) {
            ep = entry_point_;
            for (int level = max_level_; level >= 1; --level) {
                ep = greedy_search_layer(query_vec, ep, level);
            }
        }
        if (ep == INVALID_NODE || !graph_.is_alive(ep)) {
            throw std::runtime_error("Search failed: invalid entry point after finalize.");
        }

        return rabitq_search::search<D, R, BitWidth>(
            encoded, query_vec, graph_, k, gamma, visited, ep,
            calibration_.search_ip_slack_levels, calibration_.search_num_slack_levels,
            calibration_.gamma_max, calibration_.gamma_beta, calibration_.gamma_warmup);
    }

    size_t size() const { return graph_.size(); }
    size_t dim() const { return dim_; }
    bool is_finalized() const { return finalized_; }

    void save(const std::string& path) const {
        std::shared_lock<std::shared_mutex> lock(index_mutex_);
        if (!finalized_) {
            throw std::runtime_error("Index must be finalized before saving.");
        }
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot open file for writing: " + path);

        auto write_raw = [&](const void* data, size_t bytes) {
            f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
            if (!f) throw std::runtime_error("Write error: " + path);
        };

        // Magic + Version
        constexpr uint64_t magic = 0x57534E48504300ULL; // "CPHNSW\0"
        constexpr uint32_t version = 2;
        write_raw(&magic, sizeof(magic));
        write_raw(&version, sizeof(version));

        // Header
        uint32_t hdr_D = static_cast<uint32_t>(D);
        uint32_t hdr_R = static_cast<uint32_t>(R);
        uint32_t hdr_BW = static_cast<uint32_t>(BitWidth);
        uint32_t hdr_dim = static_cast<uint32_t>(dim_);
        uint64_t hdr_n = static_cast<uint64_t>(graph_.size());
        int32_t hdr_max_level = static_cast<int32_t>(max_level_);
        uint32_t hdr_ep = static_cast<uint32_t>(entry_point_);
        uint64_t hdr_seed = encoder_.get_rotation_seed();

        write_raw(&hdr_D, sizeof(hdr_D));
        write_raw(&hdr_R, sizeof(hdr_R));
        write_raw(&hdr_BW, sizeof(hdr_BW));
        write_raw(&hdr_dim, sizeof(hdr_dim));
        write_raw(&hdr_n, sizeof(hdr_n));
        write_raw(&hdr_max_level, sizeof(hdr_max_level));
        write_raw(&hdr_ep, sizeof(hdr_ep));
        write_raw(&upper_tau_, sizeof(upper_tau_));
        write_raw(&upper_alpha_, sizeof(upper_alpha_));
        write_raw(&mL_, sizeof(mL_));
        write_raw(&hdr_seed, sizeof(hdr_seed));

        // CalibrationSnapshot (POD)
        write_raw(&calibration_, sizeof(CalibrationSnapshot));

        // IndexProfile (POD)
        write_raw(&profile_, sizeof(IndexProfile));

        // Encoder centroid
        const auto& centroid = encoder_.get_centroid();
        write_raw(centroid.data(), dim_ * sizeof(float));

        // node_levels
        size_t n = static_cast<size_t>(hdr_n);
        write_raw(node_levels_.data(), n * sizeof(int));

        // Graph: norm_sq
        const auto& ns = graph_.get_norm_sq();
        write_raw(ns.data(), n * sizeof(float));

        // Graph: raw_vectors
        const auto& rv = graph_.get_raw_vectors();
        for (size_t i = 0; i < n; ++i) {
            write_raw(rv[i].data(), D * sizeof(float));
        }

        // Graph: search_data (per-element for alignment safety)
        const auto& sd = graph_.get_search_data();
        for (size_t i = 0; i < n; ++i) {
            write_raw(&sd[i], sizeof(typename Graph::SearchDataType));
        }

        // Upper layers
        uint32_t n_layers = static_cast<uint32_t>(upper_layers_.size());
        write_raw(&n_layers, sizeof(n_layers));
        for (const auto& layer : upper_layers_) {
            uint32_t layer_sz = static_cast<uint32_t>(layer.size());
            write_raw(&layer_sz, sizeof(layer_sz));
            for (const auto& edge : layer) {
                write_raw(&edge.node, sizeof(edge.node));
                uint32_t nb_cnt = static_cast<uint32_t>(edge.neighbors.size());
                write_raw(&nb_cnt, sizeof(nb_cnt));
                if (nb_cnt > 0) {
                    write_raw(edge.neighbors.data(), nb_cnt * sizeof(NodeId));
                }
            }
        }
    }

    void load(const std::string& path) {
        std::unique_lock<std::shared_mutex> lock(index_mutex_);
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot open file for reading: " + path);

        auto read_raw = [&](void* data, size_t bytes) {
            f.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(bytes));
            if (!f) throw std::runtime_error("Read error or truncated file: " + path);
        };

        // Magic
        uint64_t magic = 0;
        read_raw(&magic, sizeof(magic));
        if (magic != 0x57534E48504300ULL) {
            throw std::runtime_error("Invalid magic bytes (not a CP-HNSW index file).");
        }

        // Version
        uint32_t version = 0;
        read_raw(&version, sizeof(version));
        if (version != 2) {
            throw std::runtime_error("Unsupported index file version: " + std::to_string(version));
        }

        // Header
        uint32_t hdr_D, hdr_R, hdr_BW, hdr_dim;
        uint64_t hdr_n;
        int32_t hdr_max_level;
        uint32_t hdr_ep;
        float hdr_upper_tau, hdr_upper_alpha;
        double hdr_mL;
        uint64_t hdr_seed;

        read_raw(&hdr_D, sizeof(hdr_D));
        read_raw(&hdr_R, sizeof(hdr_R));
        read_raw(&hdr_BW, sizeof(hdr_BW));
        read_raw(&hdr_dim, sizeof(hdr_dim));
        read_raw(&hdr_n, sizeof(hdr_n));
        read_raw(&hdr_max_level, sizeof(hdr_max_level));
        read_raw(&hdr_ep, sizeof(hdr_ep));
        read_raw(&hdr_upper_tau, sizeof(hdr_upper_tau));
        read_raw(&hdr_upper_alpha, sizeof(hdr_upper_alpha));
        read_raw(&hdr_mL, sizeof(hdr_mL));
        read_raw(&hdr_seed, sizeof(hdr_seed));

        if (hdr_D != D || hdr_R != R || hdr_BW != BitWidth) {
            throw std::runtime_error(
                "Index file template parameters mismatch: file D=" +
                std::to_string(hdr_D) + " R=" + std::to_string(hdr_R) +
                " BW=" + std::to_string(hdr_BW) + ", expected D=" +
                std::to_string(D) + " R=" + std::to_string(R) +
                " BW=" + std::to_string(BitWidth));
        }
        if (hdr_dim != static_cast<uint32_t>(dim_)) {
            throw std::runtime_error(
                "Index file dim=" + std::to_string(hdr_dim) +
                " mismatches Index dim=" + std::to_string(dim_));
        }
        if (hdr_seed != encoder_.get_rotation_seed()) {
            throw std::runtime_error("Index file rotation seed mismatch.");
        }

        // CalibrationSnapshot
        CalibrationSnapshot new_calib;
        read_raw(&new_calib, sizeof(CalibrationSnapshot));

        // IndexProfile
        IndexProfile new_profile;
        read_raw(&new_profile, sizeof(IndexProfile));

        size_t n = static_cast<size_t>(hdr_n);

        // Encoder centroid
        std::vector<float> new_centroid(dim_);
        read_raw(new_centroid.data(), dim_ * sizeof(float));

        // node_levels
        std::vector<int> new_node_levels(n);
        read_raw(new_node_levels.data(), n * sizeof(int));

        // Graph: norm_sq
        AlignedVector<float> new_norm_sq(n);
        read_raw(new_norm_sq.data(), n * sizeof(float));

        // Graph: raw_vectors
        using RawVector = typename Graph::RawVector;
        std::vector<RawVector> new_raw_vectors(n);
        for (size_t i = 0; i < n; ++i) {
            read_raw(new_raw_vectors[i].data(), D * sizeof(float));
        }

        // Graph: search_data
        using SearchDataType = typename Graph::SearchDataType;
        std::vector<SearchDataType, AlignedAllocator<SearchDataType>> new_search_data(n);
        for (size_t i = 0; i < n; ++i) {
            read_raw(&new_search_data[i], sizeof(SearchDataType));
        }

        // Upper layers
        uint32_t n_layers;
        read_raw(&n_layers, sizeof(n_layers));
        std::vector<std::vector<HNSWLayerEdge>> new_upper_layers(n_layers);
        for (uint32_t l = 0; l < n_layers; ++l) {
            uint32_t layer_sz;
            read_raw(&layer_sz, sizeof(layer_sz));
            new_upper_layers[l].resize(layer_sz);
            for (uint32_t e = 0; e < layer_sz; ++e) {
                read_raw(&new_upper_layers[l][e].node, sizeof(NodeId));
                uint32_t nb_cnt;
                read_raw(&nb_cnt, sizeof(nb_cnt));
                new_upper_layers[l][e].neighbors.resize(nb_cnt);
                if (nb_cnt > 0) {
                    read_raw(new_upper_layers[l][e].neighbors.data(),
                             nb_cnt * sizeof(NodeId));
                }
            }
        }

        // All reads succeeded — commit state
        graph_.restore_from_serialized(
            std::move(new_search_data),
            std::move(new_raw_vectors),
            std::move(new_norm_sq),
            static_cast<NodeId>(hdr_ep));

        encoder_.set_centroid(std::move(new_centroid));
        calibration_ = new_calib;
        profile_ = new_profile;
        node_levels_ = std::move(new_node_levels);
        upper_layers_ = std::move(new_upper_layers);
        max_level_ = static_cast<int>(hdr_max_level);
        entry_point_ = static_cast<NodeId>(hdr_ep);
        upper_tau_ = hdr_upper_tau;
        upper_alpha_ = hdr_upper_alpha;
        mL_ = hdr_mL;

        finalized_ = true;
        needs_build_ = false;
    }

private:
    size_t dim_;
    Encoder encoder_;
    Graph graph_;
    bool finalized_ = false;
    bool needs_build_ = false;

    CalibrationSnapshot calibration_;
    IndexProfile profile_;

    double mL_;
    std::mt19937_64 rng_;

    int max_level_ = 0;
    NodeId entry_point_ = INVALID_NODE;
    float upper_tau_ = 0.0f;
    float upper_alpha_ = 1.2f;

    std::vector<int> node_levels_;

    std::vector<std::vector<HNSWLayerEdge>> upper_layers_;
    mutable std::shared_mutex index_mutex_;

    const HNSWLayerEdge* find_edge(int level, NodeId node) const {
        const auto& layer = upper_layers_[level - 1];
        HNSWLayerEdge target{node, {}};
        auto it = std::lower_bound(layer.begin(), layer.end(), target);
        if (it != layer.end() && it->node == node) return &(*it);
        return nullptr;
    }

    HNSWLayerEdge& get_or_create_edge(int level, NodeId node) {
        auto& layer = upper_layers_[level - 1];
        HNSWLayerEdge target{node, {}};
        auto it = std::lower_bound(layer.begin(), layer.end(), target);
        if (it != layer.end() && it->node == node) return *it;
        return *layer.insert(it, HNSWLayerEdge{node, {}});
    }

    void assign_layers(size_t n) {
        node_levels_.resize(n);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        max_level_ = 0;
        entry_point_ = INVALID_NODE;

        for (size_t i = 0; i < n; ++i) {
            double r = dist(rng_);
            if (r < constants::kMinLayerRandom) r = constants::kMinLayerRandom;
            int level = static_cast<int>(-std::log(r) * mL_);
            node_levels_[i] = level;
            if (entry_point_ == INVALID_NODE || level > max_level_) {
                max_level_ = level;
                entry_point_ = static_cast<NodeId>(i);
            }
        }

        upper_layers_.resize(max_level_);
    }

    void build_upper_layers() {
        size_t n = graph_.size();

        std::vector<NodeId> insertion_order(n);
        for (size_t i = 0; i < n; ++i) insertion_order[i] = static_cast<NodeId>(i);
        std::sort(insertion_order.begin(), insertion_order.end(),
                  [this](NodeId a, NodeId b) { return node_levels_[a] > node_levels_[b]; });

        // Count upper-layer nodes
        size_t n_upper = 0;
        for (size_t i = 0; i < n; ++i) {
            if (node_levels_[insertion_order[i]] > 0) n_upper++;
            else break;
        }

        // Data-adaptive sample sizes: 10·√n_upper balances cost (O(n_upper·samples))
        // with statistical coverage (CLT: ≥√n samples for stable median estimation)
        size_t dist_samples = std::min(
            static_cast<size_t>(std::sqrt(static_cast<float>(n_upper)) * 10.0f),
            n_upper);
        size_t nn_limit = std::min(dist_samples * 2, n_upper);

        std::vector<float> upper_nn_dists;
        for (size_t idx = 0; idx < n && upper_nn_dists.size() < dist_samples; ++idx) {
            NodeId node = insertion_order[idx];
            if (node_levels_[node] == 0) break;
            float best = std::numeric_limits<float>::max();
            for (size_t jdx = 0; jdx < n && jdx < nn_limit; ++jdx) {
                NodeId other = insertion_order[jdx];
                if (other == node) continue;
                if (node_levels_[other] == 0) break;
                float d = l2_distance_simd<D>(graph_.get_vector(node), graph_.get_vector(other));
                if (d < best) best = d;
            }
            if (best < std::numeric_limits<float>::max()) {
                upper_nn_dists.push_back(best);
            }
        }
        if (!upper_nn_dists.empty()) {
            std::sort(upper_nn_dists.begin(), upper_nn_dists.end());
            // Tau: MAD-based robust standard deviation of upper NN distances
            float nn_median = upper_nn_dists[upper_nn_dists.size() / 2];
            std::vector<float> abs_devs(upper_nn_dists.size());
            for (size_t i = 0; i < upper_nn_dists.size(); ++i)
                abs_devs[i] = std::abs(upper_nn_dists[i] - nn_median);
            std::sort(abs_devs.begin(), abs_devs.end());
            float mad = abs_devs[abs_devs.size() / 2];
            upper_tau_ = constants::kMadNormFactor * mad;

            // Alpha: derive from NN distance spread (CV + 1)
            float mean_dist = 0.0f;
            for (float d : upper_nn_dists) mean_dist += d;
            mean_dist /= static_cast<float>(upper_nn_dists.size());
            float var_dist = 0.0f;
            for (float d : upper_nn_dists) var_dist += (d - mean_dist) * (d - mean_dist);
            var_dist /= static_cast<float>(upper_nn_dists.size());
            float cv = (mean_dist > constants::eps::kSmall)
                ? std::sqrt(var_dist) / mean_dist : 0.2f;
            upper_alpha_ = 1.0f + cv;
        }

        for (size_t idx = 0; idx < n; ++idx) {
            NodeId node = insertion_order[idx];
            int node_level = node_levels_[node];
            if (node_level == 0) break;

            NodeId ep = entry_point_;
            for (int level = max_level_; level > node_level; --level) {
                ep = greedy_search_layer(graph_.get_vector(node), ep, level);
            }

            for (int level = std::min(node_level, max_level_); level >= 1; --level) {
                // Layer-aware ef: grows with level, capped at 4*R
                size_t upper_ef = std::clamp(
                    static_cast<size_t>(static_cast<float>(R) *
                        (1.0f + static_cast<float>(level) *
                         std::log(static_cast<float>(std::max(n_upper, size_t(2)))) /
                         std::log(static_cast<float>(std::max(n, size_t(2)))))),
                    R, R * 4);
                auto candidates = search_upper_layer(
                    graph_.get_vector(node), ep, level, upper_ef);

                auto dist_fn = [this](NodeId a, NodeId b) {
                    return l2_distance_simd<D>(graph_.get_vector(a), graph_.get_vector(b));
                };
                auto selected = select_neighbors_alpha_cng(
                    std::move(candidates), M_UPPER, dist_fn, [](NodeId) { return 0.0f; },
                    upper_alpha_, upper_tau_);

                auto& node_neighbors = get_or_create_edge(level, node).neighbors;
                node_neighbors.clear();
                node_neighbors.reserve(selected.size());
                for (const auto& s : selected) {
                    node_neighbors.push_back(s.id);
                }

                for (const auto& s : selected) {
                    auto& nb = get_or_create_edge(level, s.id).neighbors;
                    nb.push_back(node);
                    if (nb.size() > M_UPPER) {
                        prune_upper_neighbors(s.id, level);
                    }
                }

                if (!selected.empty()) {
                    ep = selected[0].id;
                }
            }
        }

    }

    NodeId greedy_search_layer(const float* query, NodeId ep, int level) const {
        float best_dist = l2_distance_simd<D>(query, graph_.get_vector(ep));
        NodeId best_id = ep;
        bool improved = true;

        while (improved) {
            improved = false;
            const auto* edge = find_edge(level, best_id);
            if (!edge) break;
            const auto& neighbors = edge->neighbors;
            for (NodeId nb : neighbors) {
                float d = l2_distance_simd<D>(query, graph_.get_vector(nb));
                if (d < best_dist) {
                    best_dist = d;
                    best_id = nb;
                    improved = true;
                }
            }
        }

        return best_id;
    }

    std::vector<NeighborCandidate> search_upper_layer(
        const float* query, NodeId ep, int level, size_t ef) const
    {
        MinHeap candidates;
        MaxHeap nearest;

        float ep_dist = l2_distance_simd<D>(query, graph_.get_vector(ep));
        candidates.push({ep, ep_dist});
        nearest.push({ep, ep_dist});

        thread_local VisitationTable visited_table(0);
        if (visited_table.capacity() < graph_.size()) {
            visited_table.resize(graph_.size() + adaptive_defaults::visitation_headroom(graph_.size()));
        }
        uint64_t qid = visited_table.new_query();
        visited_table.check_and_mark(ep, qid);

        while (!candidates.empty()) {
            auto current = candidates.top();
            candidates.pop();

            if (nearest.size() >= ef && current.distance > nearest.top().distance) {
                break;
            }

            const auto* edge = find_edge(level, current.id);
            if (!edge) continue;
            const auto& neighbors = edge->neighbors;
            for (NodeId nb : neighbors) {
                if (visited_table.check_and_mark(nb, qid)) continue;

                float d = l2_distance_simd<D>(query, graph_.get_vector(nb));

                if (nearest.size() < ef || d < nearest.top().distance) {
                    candidates.push({nb, d});
                    nearest.push({nb, d});
                    if (nearest.size() > ef) {
                        nearest.pop();
                    }
                }
            }
        }

        std::vector<NeighborCandidate> results;
        results.reserve(nearest.size());
        while (!nearest.empty()) {
            results.push_back({nearest.top().id, nearest.top().distance});
            nearest.pop();
        }
        std::sort(results.begin(), results.end());
        return results;
    }

    void prune_upper_neighbors(NodeId node, int level) {
        auto& nb = get_or_create_edge(level, node).neighbors;
        if (nb.size() <= M_UPPER) return;

        const float* vec = graph_.get_vector(node);
        std::vector<NeighborCandidate> candidates;
        candidates.reserve(nb.size());
        for (NodeId id : nb) {
            float d = l2_distance_simd<D>(vec, graph_.get_vector(id));
            candidates.push_back({id, d});
        }

        auto dist_fn = [this](NodeId a, NodeId b) {
            return l2_distance_simd<D>(graph_.get_vector(a), graph_.get_vector(b));
        };
        auto selected = select_neighbors_alpha_cng(
            std::move(candidates), M_UPPER, dist_fn, [](NodeId) { return 0.0f; },
            upper_alpha_, upper_tau_);
        nb.clear();
        nb.reserve(selected.size());
        for (const auto& s : selected) {
            nb.push_back(s.id);
        }
    }

    void calibrate_estimator(size_t num_samples) {
        size_t n = graph_.size();
        if (n < constants::kMinCalibrateNodes) {
            throw std::runtime_error("Calibration requires at least 50 nodes.");
        }

        std::vector<NodeId> sample_ids(n);
        for (size_t i = 0; i < n; ++i) {
            sample_ids[i] = static_cast<NodeId>(i);
        }

        std::mt19937 rng(static_cast<uint32_t>(constants::kDefaultLayerSeed + constants::kDefaultCalibrationSeed));
        std::shuffle(sample_ids.begin(), sample_ids.end(), rng);

        size_t n_db = std::min(num_samples, n);
        size_t n_synth = std::min(num_samples / 2, n);

        std::vector<float> dim_var(D, 0.0f);
        size_t var_sample = std::min(n, num_samples / 4);
        for (size_t i = 0; i < var_sample; ++i) {
            const float* v = graph_.get_vector(sample_ids[i]);
            for (size_t d = 0; d < D; ++d) {
                dim_var[d] += v[d] * v[d];
            }
        }
        std::vector<float> dim_mean(D, 0.0f);
        for (size_t i = 0; i < var_sample; ++i) {
            const float* v = graph_.get_vector(sample_ids[i]);
            for (size_t d = 0; d < D; ++d) {
                dim_mean[d] += v[d];
            }
        }
        for (size_t d = 0; d < D; ++d) {
            dim_mean[d] /= static_cast<float>(var_sample);
            dim_var[d] = dim_var[d] / static_cast<float>(var_sample) - dim_mean[d] * dim_mean[d];
            if (dim_var[d] < constants::eps::kSmall) dim_var[d] = constants::eps::kSmall;
        }


        struct CalibSample {
            float nop;
            float ip_corrected;
            float ip_qo_denom;
            float dist_qp_sq;
            NodeId neighbor;
            size_t query_idx;
        };

        std::vector<float> ip_qo_values;
        std::vector<float> per_sample_ip_corrected;
        std::vector<float> per_sample_ip_qo;
        std::vector<float> truths;
        std::vector<float> nn_dists_sq;
        std::vector<float> nop_samples;
        std::vector<CalibSample> calib_samples;


        std::vector<AlignedVector<float>> query_buffer;
        query_buffer.reserve(n_db + n_synth);

        ip_qo_values.reserve(num_samples * 4);
        per_sample_ip_corrected.reserve(num_samples * 4);
        per_sample_ip_qo.reserve(num_samples * 4);
        truths.reserve(num_samples * 4);
        nop_samples.reserve(num_samples * 4);
        calib_samples.reserve(num_samples * 4);

        size_t parent_cursor = 0;
        auto process_query = [&](const float* query_vec, size_t query_idx) {
            NodeId parent = sample_ids[parent_cursor % n];
            parent_cursor++;

            float best_dist = l2_distance_simd<D>(query_vec, graph_.get_vector(parent));
            const auto& nb = graph_.get_neighbors(parent);

            for (size_t i = 0; i < nb.size(); ++i) {
                NodeId nid = nb.neighbor_ids[i];
                if (nid == INVALID_NODE) break;
                float d = l2_distance_simd<D>(query_vec, graph_.get_vector(nid));
                if (d < best_dist) {
                    best_dist = d;
                    parent = nid;
                }
            }
            nn_dists_sq.push_back(best_dist);

            const auto& pnb = graph_.get_neighbors(parent);

            QueryType encoded = encoder_.encode_query_raw(query_vec);

            float dist_qp_sq = l2_distance_simd<D>(query_vec, graph_.get_vector(parent));

            size_t num_batches = (pnb.size() + constants::kFastScanBatch - 1) / constants::kFastScanBatch;
            for (size_t batch = 0; batch < num_batches; ++batch) {
                size_t batch_start = batch * constants::kFastScanBatch;
                size_t batch_count = std::min(constants::kFastScanBatch, pnb.size() - batch_start);

                alignas(64) uint32_t fastscan_sums[constants::kFastScanBatch];
                if constexpr (BitWidth == 1) {
                    fastscan::compute_inner_products(
                        encoded.lut, pnb.code_blocks[batch], fastscan_sums);
                } else {
                    alignas(64) uint32_t msb_sums[constants::kFastScanBatch];
                    fastscan::compute_nbit_inner_products<D, BitWidth>(
                        encoded.lut, pnb.code_blocks[batch],
                        fastscan_sums, msb_sums);
                }

                for (size_t j = 0; j < batch_count; ++j) {
                    size_t ni = batch_start + j;
                    NodeId neighbor = pnb.neighbor_ids[ni];
                    if (neighbor == INVALID_NODE) break;

                    float ip_qo = pnb.ip_qo[ni];
                    ip_qo_values.push_back(ip_qo);

                    float nop = std::max(pnb.nop[ni], constants::eps::kSmall);
                    nop_samples.push_back(nop);

                    float A = encoded.coeff_fastscan;
                    float B = encoded.coeff_popcount;
                    float C = encoded.coeff_constant;

                    float ip_approx;
                    if constexpr (BitWidth == 1) {
                        ip_approx = A * static_cast<float>(fastscan_sums[j])
                                  + B * static_cast<float>(pnb.popcounts[ni]) + C;
                    } else {
                        constexpr float K = static_cast<float>((1u << BitWidth) - 1);
                        constexpr float inv_K = 1.0f / K;
                        ip_approx = A * inv_K * static_cast<float>(fastscan_sums[j])
                                  + B * inv_K * static_cast<float>(pnb.weighted_popcounts[ni]) + C;
                    }

                    float ip_corrected = ip_approx - pnb.ip_cp[ni];
                    float ip_qo_denom = std::max(std::abs(ip_qo), constants::eps::kMedium);

                    const float* p_vec = graph_.get_vector(parent);
                    const float* o_vec = graph_.get_vector(neighbor);
                    float true_ip = 0.0f;
                    for (size_t d = 0; d < D; ++d) {
                        true_ip += (query_vec[d] - p_vec[d]) * (o_vec[d] - p_vec[d]);
                    }
                    true_ip /= nop;

                    per_sample_ip_corrected.push_back(ip_corrected);
                    per_sample_ip_qo.push_back(ip_qo_denom);
                    truths.push_back(true_ip);

                    calib_samples.push_back({nop, ip_corrected, ip_qo_denom,
                                             dist_qp_sq, neighbor, query_idx});
                }
            }
        };

        for (size_t i = 0; i < n_db; ++i) {
            const float* v = graph_.get_vector(sample_ids[i]);
            AlignedVector<float> qbuf(D);
            std::memcpy(qbuf.data(), v, D * sizeof(float));
            query_buffer.push_back(std::move(qbuf));
            process_query(v, query_buffer.size() - 1);
        }

        std::normal_distribution<float> normal_dist(0.0f, 1.0f);
        for (size_t i = 0; i < n_synth; ++i) {
            const float* base = graph_.get_vector(sample_ids[i % n]);
            AlignedVector<float> synth_query(D);
            for (size_t d = 0; d < D; ++d) {
                synth_query[d] = base[d] + normal_dist(rng) * std::sqrt(dim_var[d]);
            }
            query_buffer.push_back(synth_query);
            process_query(query_buffer.back().data(), query_buffer.size() - 1);
        }

        if (ip_qo_values.empty()) {
            throw std::runtime_error("Calibration failed: no ip_qo samples.");
        }

        // MAD-based robust lower fence for ip_qo_floor
        std::sort(ip_qo_values.begin(), ip_qo_values.end());
        float median_ipqo = ip_qo_values[ip_qo_values.size() / 2];
        {
            std::vector<float> abs_devs(ip_qo_values.size());
            for (size_t i = 0; i < ip_qo_values.size(); ++i)
                abs_devs[i] = std::abs(ip_qo_values[i] - median_ipqo);
            std::sort(abs_devs.begin(), abs_devs.end());
            float mad = abs_devs[abs_devs.size() / 2];
            float sigma_est = constants::kMadNormFactor * mad;
            // 3-sigma lower fence: standard normal outlier threshold
            calibration_.ip_qo_floor = std::max(
                median_ipqo - 3.0f * sigma_est,  // 3-sigma lower fence
                constants::eps::kMedium);
        }

        std::vector<float> floored_estimates;
        floored_estimates.reserve(per_sample_ip_corrected.size());
        for (size_t i = 0; i < per_sample_ip_corrected.size(); ++i) {
            float floored_qo = std::max(per_sample_ip_qo[i], calibration_.ip_qo_floor);
            floored_estimates.push_back(per_sample_ip_corrected[i] / floored_qo);
        }

        if (floored_estimates.size() < 20) {
            throw std::runtime_error("Calibration failed: too few estimator/target pairs.");
        }


        size_t np = floored_estimates.size();
        double sum_e = 0, sum_t = 0, sum_ee = 0, sum_et = 0;
        for (size_t i = 0; i < np; ++i) {
            double e = floored_estimates[i];
            double t = truths[i];
            sum_e += e;
            sum_t += t;
            sum_ee += e * e;
            sum_et += e * t;
        }
        double mean_e = sum_e / np;
        double mean_t = sum_t / np;
        double var_e = sum_ee / np - mean_e * mean_e;
        double cov_et = sum_et / np - mean_e * mean_t;

        double a = 1.0, b = 0.0;
        if (var_e > constants::eps::kSmall) {
            a = cov_et / var_e;
            b = mean_t - a * mean_e;
        }

        std::vector<float> abs_residuals(np);
        for (int iter = 0; iter < constants::kHuberMaxIter; ++iter) {
            for (size_t i = 0; i < np; ++i) {
                float r = truths[i] - static_cast<float>(a * floored_estimates[i] + b);
                abs_residuals[i] = std::abs(r);
            }
            std::sort(abs_residuals.begin(), abs_residuals.end());
            float mad = abs_residuals[np / 2];
            float huber_delta = constants::kHuberDeltaScale * constants::kMadNormFactor * mad;
            if (huber_delta < constants::eps::kSmall) break;

            double wsum_e = 0, wsum_t = 0, wsum_ee = 0, wsum_et = 0, wsum = 0;
            for (size_t i = 0; i < np; ++i) {
                float r = truths[i] - static_cast<float>(a * floored_estimates[i] + b);
                float ar = std::abs(r);
                float w = (ar <= huber_delta) ? 1.0f : huber_delta / ar;
                double wd = w;
                double e = floored_estimates[i];
                double t = truths[i];
                wsum += wd;
                wsum_e += wd * e;
                wsum_t += wd * t;
                wsum_ee += wd * e * e;
                wsum_et += wd * e * t;
            }
            double wm_e = wsum_e / wsum;
            double wm_t = wsum_t / wsum;
            double wvar = wsum_ee / wsum - wm_e * wm_e;
            double wcov = wsum_et / wsum - wm_e * wm_t;
            if (wvar > constants::eps::kSmall) {
                double a_new = wcov / wvar;
                double b_new = wm_t - a_new * wm_e;
                if (std::abs(a_new - a) + std::abs(b_new - b) < constants::kHuberConvergeTol) {
                    a = a_new;
                    b = b_new;
                    break;
                }
                a = a_new;
                b = b_new;
            }
        }


        // Degenerate regression detection via R² and leverage diagnostics
        double ss_res = 0.0, ss_tot = 0.0;
        for (size_t i = 0; i < np; ++i) {
            double fitted = a * floored_estimates[i] + b;
            double residual = truths[i] - fitted;
            ss_res += residual * residual;
            ss_tot += (truths[i] - mean_t) * (truths[i] - mean_t);
        }
        float r_squared = (ss_tot > constants::eps::kSmall)
            ? static_cast<float>(1.0 - ss_res / ss_tot) : 0.0f;

        // Max leverage (hat matrix diagonal): h_i = 1/n + (x_i - x_bar)² / SXX
        double sxx = var_e * static_cast<double>(np);
        float max_leverage = 0.0f;
        if (sxx > constants::eps::kSmall) {
            for (size_t i = 0; i < np; ++i) {
                double h = 1.0 / np + (floored_estimates[i] - mean_e)
                                      * (floored_estimates[i] - mean_e) / sxx;
                max_leverage = std::max(max_leverage, static_cast<float>(h));
            }
        }
        // Degenerate if R² < 0.1 (standard threshold: < 10% variance explained)
        // or max leverage > 2p/n (Cook's criterion, p=2 for intercept+slope)
        float leverage_threshold = 4.0f / static_cast<float>(std::max(np, size_t(1)));
        bool degenerate = (r_squared < 0.1f) || (max_leverage > leverage_threshold);
        if (degenerate) {
            a = 1.0; b = 0.0;
        }
        calibration_.affine_a = static_cast<float>(a);
        calibration_.affine_b = static_cast<float>(b);

        if (nn_dists_sq.empty()) {
            throw std::runtime_error("Calibration failed: no nearest-neighbor distance samples.");
        }
        std::sort(nn_dists_sq.begin(), nn_dists_sq.end());
        calibration_.median_nn_dist_sq = nn_dists_sq[nn_dists_sq.size() / 2];
        calibration_.min_slack_sq = std::max(
            constants::eps::kSmall,
            calibration_.median_nn_dist_sq * 1e-4f);


        std::vector<float> dist_residuals;
        dist_residuals.reserve(calib_samples.size());
        for (const auto& s : calib_samples) {
            float floored_qo = std::max(s.ip_qo_denom, calibration_.ip_qo_floor);
            float ip_est = (floored_qo > constants::eps::kMedium)
                ? s.ip_corrected / floored_qo : 0.0f;
            ip_est = calibration_.affine_a * ip_est + calibration_.affine_b;
            float est_dist = std::max(s.nop * s.nop + s.dist_qp_sq
                             - 2.0f * s.nop * ip_est, 0.0f);
            const float* qvec = query_buffer[s.query_idx].data();
            float true_dist = l2_distance_simd<D>(qvec, graph_.get_vector(s.neighbor));
            dist_residuals.push_back(std::abs(est_dist - true_dist));
        }

        std::sort(dist_residuals.begin(), dist_residuals.end());


        // EVT threshold bounds derived from sample size
        size_t n_resid = dist_residuals.size();
        float evt_thresh_min = std::max(
            1.0f - 1.0f / std::sqrt(static_cast<float>(std::max(n_resid, size_t(4)))),
            0.5f);
        float evt_thresh_max = 1.0f - static_cast<float>(profile_.evt_min_tail) /
            static_cast<float>(std::max(n_resid, size_t(1)));

        calibration_.evt = evt_crc::fit_gpd_stable(
            dist_residuals.data(), n_resid, profile_.evt_min_tail,
            evt_thresh_min, evt_thresh_max);

        if (nop_samples.empty()) {
            throw std::runtime_error("Calibration failed: no nop samples.");
        }
        std::sort(nop_samples.begin(), nop_samples.end());
        calibration_.median_nop = nop_samples[nop_samples.size() / 2];

        if (!calibration_.evt.fitted || calibration_.median_nop <= 0.0f) {
            throw std::runtime_error("Calibration failed: EVT-CRC fit did not converge.");
        }


        // Gamma bounds from IQR-based Tukey fences on normalized residuals
        float ref = std::sqrt(std::max(calibration_.median_nn_dist_sq, calibration_.min_slack_sq));
        std::vector<float> norm_resid(n_resid);
        for (size_t i = 0; i < n_resid; ++i) {
            norm_resid[i] = dist_residuals[i] / ref;
        }
        // Already sorted (dist_residuals sorted and ref > 0)
        float nr_q1 = norm_resid[n_resid / 4];
        float nr_median = norm_resid[n_resid / 2];
        float nr_q3 = norm_resid[3 * n_resid / 4];
        float nr_iqr = nr_q3 - nr_q1;

        // gamma_min_floor: 1 + 1/√D (RaBitQ quantization noise floor)
        float gamma_min_floor = 1.0f + 1.0f / std::sqrt(static_cast<float>(D));
        float gamma_min_data = 1.0f + norm_resid[std::max(size_t(1), n_resid / 100)];
        calibration_.gamma_min = std::max(gamma_min_data, gamma_min_floor);

        // gamma_max: Tukey upper fence = 1 + Q3 + 1.5*IQR
        float gamma_max_fence = 1.0f + nr_q3 + 1.5f * nr_iqr;
        // Minimum offset: at least IQR or median of normalized residuals
        float min_offset = std::max(nr_iqr, nr_median);
        calibration_.gamma_max = std::max(gamma_max_fence,
            calibration_.gamma_min + min_offset);


        // Gamma beta: 1/CV with floor at CV estimator resolution
        double rmean = 0.0;
        for (float r : dist_residuals) rmean += r;
        rmean /= static_cast<double>(n_resid);
        double rvar = 0.0;
        for (float r : dist_residuals) rvar += (r - rmean) * (r - rmean);
        rvar /= static_cast<double>(n_resid);
        float resid_cv = static_cast<float>(
            std::sqrt(rvar) / std::max(rmean, static_cast<double>(constants::eps::kSmall)));
        // CV floor = standard error of the CV estimator: 1/√(2·(n-1))
        float cv_floor = 1.0f / std::sqrt(
            2.0f * static_cast<float>(std::max(n_resid, size_t(2)) - 1));
        calibration_.gamma_beta = 1.0f / std::max(resid_cv, cv_floor);

        // Warmup = √(n_tail): need enough ratio observations for stable variance estimate.
        // Floor of 4: minimum for 2nd-moment estimation.
        calibration_.gamma_warmup = std::max(size_t(4),
            static_cast<size_t>(std::ceil(std::sqrt(static_cast<float>(calibration_.evt.n_tail)))));


        calibration_.slack_levels = profile_.slack_levels;

        // Precompute max-recall search parameters (internal delta = 1e-4)
        constexpr float kSearchDelta = 1e-4f;
        float delta_prune = 0.5f * kSearchDelta;
        float delta_term = 0.5f * kSearchDelta;

        int evt_L = std::clamp(calibration_.slack_levels, 1, constants::kMaxSlackArray);
        calibration_.search_num_slack_levels = evt_L;

        constexpr float BASEL_K = constants::kBaselK;
        for (int i = 1; i <= evt_L; ++i) {
            float i_f = static_cast<float>(i);
            float alpha_i = delta_prune * BASEL_K / (i_f * i_f);
            float dist_slack = evt_crc::evt_quantile(alpha_i, calibration_.evt);
            calibration_.search_ip_slack_levels[i - 1] = dist_slack
                / (2.0f * calibration_.median_nop);
        }

        float alpha_term = delta_term;
        float dist_slack_term = evt_crc::evt_quantile(alpha_term, calibration_.evt);
        calibration_.search_gamma = std::clamp(
            1.0f + dist_slack_term / ref,
            calibration_.gamma_min, calibration_.gamma_max);

    }
};

}
