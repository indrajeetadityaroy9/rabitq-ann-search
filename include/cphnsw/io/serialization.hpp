#pragma once

#include "../graph/rabitq_graph.hpp"
#include "../core/evt_crc.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace cphnsw {

struct SerializationHeader {
    char magic[8] = {'R','B','Q','G','R','P','H', '\0'};
    uint32_t dims;
    uint32_t degree;
    uint32_t bit_width;
    uint64_t num_nodes;
    uint32_t entry_point;
    uint32_t original_dim;
};

struct CalibrationSnapshot {
    float affine_a = 1.0f;
    float affine_b = 0.0f;
    float ip_qo_floor = 0.0f;
    float resid_q99_dot = 0.0f;
    float resid_sigma = 0.0f;
    float median_nn_dist_sq = 0.0f;
    float calibration_corr = 0.0f;
    uint32_t flags = 0;
    EVTState evt;
};

struct HNSWLayerEdge {
    NodeId node;
    std::vector<NodeId> neighbors;
};

struct HNSWLayerSnapshot {
    int max_level = 0;
    NodeId entry_point = INVALID_NODE;
    float upper_tau = 0.0f;
    std::vector<int> node_levels;
    std::vector<std::vector<HNSWLayerEdge>> upper_layers;
};

template <size_t D, size_t R = 32, size_t BitWidth = 1>
class IndexSerializer {
public:
    using Graph = RaBitQGraph<D, R, BitWidth>;
    using NeighborBlockType = typename Graph::NeighborBlockType;

    static void save(const std::string& path, const Graph& graph,
                     const HNSWLayerSnapshot& hnsw_data,
                     const CalibrationSnapshot& cal = CalibrationSnapshot()) {
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) throw std::runtime_error("Cannot open file for writing: " + path);

        SerializationHeader header;
        header.dims = static_cast<uint32_t>(D);
        header.degree = static_cast<uint32_t>(R);
        header.bit_width = static_cast<uint32_t>(BitWidth);
        header.num_nodes = graph.size();
        header.entry_point = graph.entry_point();
        header.original_dim = static_cast<uint32_t>(graph.dim());

        std::fwrite(&header, sizeof(header), 1, f);

        for (size_t i = 0; i < graph.size(); ++i) {
            NodeId id = static_cast<NodeId>(i);
            const auto& code = graph.get_code(id);
            std::fwrite(&code, sizeof(code), 1, f);

            const auto& nb = graph.get_neighbors(id);
            std::fwrite(&nb, sizeof(nb), 1, f);

            const float* vec = graph.get_vector(id);
            std::fwrite(vec, sizeof(float), D, f);
        }

        write_hnsw_layers(f, hnsw_data, graph.size());

        std::fwrite(&cal.affine_a, sizeof(float), 1, f);
        std::fwrite(&cal.affine_b, sizeof(float), 1, f);
        std::fwrite(&cal.ip_qo_floor, sizeof(float), 1, f);
        std::fwrite(&cal.resid_q99_dot, sizeof(float), 1, f);
        std::fwrite(&cal.resid_sigma, sizeof(float), 1, f);
        std::fwrite(&cal.median_nn_dist_sq, sizeof(float), 1, f);
        std::fwrite(&cal.calibration_corr, sizeof(float), 1, f);
        std::fwrite(&cal.flags, sizeof(uint32_t), 1, f);

        uint8_t evt_fitted = cal.evt.fitted ? 1 : 0;
        std::fwrite(&evt_fitted, sizeof(uint8_t), 1, f);
        std::fwrite(&cal.evt.u, sizeof(float), 1, f);
        std::fwrite(&cal.evt.p_u, sizeof(float), 1, f);
        std::fwrite(&cal.evt.xi, sizeof(float), 1, f);
        std::fwrite(&cal.evt.beta, sizeof(float), 1, f);
        std::fwrite(&cal.evt.nop_p95, sizeof(float), 1, f);
        std::fwrite(&cal.evt.n_resid, sizeof(uint32_t), 1, f);
        std::fwrite(&cal.evt.n_tail, sizeof(uint32_t), 1, f);

        std::fclose(f);
    }

    struct LoadResult {
        Graph graph;
        HNSWLayerSnapshot hnsw_data;
        CalibrationSnapshot calibration;
    };

    static LoadResult load(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("Cannot open file for reading: " + path);

        SerializationHeader header;
        if (std::fread(&header, sizeof(header), 1, f) != 1) {
            std::fclose(f);
            throw std::runtime_error("Failed to read header");
        }

        if (std::memcmp(header.magic, "RBQGRPH", 7) != 0) {
            std::fclose(f);
            throw std::runtime_error("Invalid magic number");
        }

        if (header.dims != D || header.degree != R || header.bit_width != BitWidth) {
            std::fclose(f);
            throw std::runtime_error("Template parameter mismatch");
        }

        Graph graph(header.original_dim, header.num_nodes);

        using CodeType = typename Graph::CodeType;

        for (uint64_t i = 0; i < header.num_nodes; ++i) {
            CodeType code;
            if (std::fread(&code, sizeof(code), 1, f) != 1) {
                std::fclose(f);
                throw std::runtime_error("Failed to read code");
            }

            NeighborBlockType nb;
            if (std::fread(&nb, sizeof(nb), 1, f) != 1) {
                std::fclose(f);
                throw std::runtime_error("Failed to read neighbors");
            }

            float vec[D];
            if (std::fread(vec, sizeof(float), D, f) != D) {
                std::fclose(f);
                throw std::runtime_error("Failed to read vector");
            }

            NodeId id = graph.add_node(code, vec);
            graph.get_neighbors(id) = nb;
        }

        graph.set_entry_point(static_cast<NodeId>(header.entry_point));

        HNSWLayerSnapshot hnsw_data;
        read_hnsw_layers(f, hnsw_data, header.num_nodes);

        CalibrationSnapshot cal;
        if (std::fread(&cal.affine_a, sizeof(float), 1, f) != 1 ||
            std::fread(&cal.affine_b, sizeof(float), 1, f) != 1 ||
            std::fread(&cal.ip_qo_floor, sizeof(float), 1, f) != 1 ||
            std::fread(&cal.resid_q99_dot, sizeof(float), 1, f) != 1 ||
            std::fread(&cal.resid_sigma, sizeof(float), 1, f) != 1 ||
            std::fread(&cal.median_nn_dist_sq, sizeof(float), 1, f) != 1 ||
            std::fread(&cal.calibration_corr, sizeof(float), 1, f) != 1 ||
            std::fread(&cal.flags, sizeof(uint32_t), 1, f) != 1)
            throw std::runtime_error("Failed to read calibration state");

        uint8_t evt_fitted = 0;
        if (std::fread(&evt_fitted, sizeof(uint8_t), 1, f) != 1)
            throw std::runtime_error("Failed to read EVT-CRC state");
        cal.evt.fitted = (evt_fitted != 0);
        std::fread(&cal.evt.u, sizeof(float), 1, f);
        std::fread(&cal.evt.p_u, sizeof(float), 1, f);
        std::fread(&cal.evt.xi, sizeof(float), 1, f);
        std::fread(&cal.evt.beta, sizeof(float), 1, f);
        std::fread(&cal.evt.nop_p95, sizeof(float), 1, f);
        std::fread(&cal.evt.n_resid, sizeof(uint32_t), 1, f);
        std::fread(&cal.evt.n_tail, sizeof(uint32_t), 1, f);

        std::fclose(f);

        graph.recompute_norms();

        return {std::move(graph), std::move(hnsw_data), cal};
    }

private:
    static void write_hnsw_layers(FILE* f, const HNSWLayerSnapshot& data, size_t num_nodes) {
        std::fwrite(&data.max_level, sizeof(int), 1, f);
        std::fwrite(&data.entry_point, sizeof(NodeId), 1, f);
        std::fwrite(&data.upper_tau, sizeof(float), 1, f);

        uint64_t levels_size = data.node_levels.size();
        std::fwrite(&levels_size, sizeof(uint64_t), 1, f);
        if (levels_size > 0) {
            std::fwrite(data.node_levels.data(), sizeof(int), levels_size, f);
        }

        uint32_t num_layers = static_cast<uint32_t>(data.upper_layers.size());
        std::fwrite(&num_layers, sizeof(uint32_t), 1, f);

        for (uint32_t layer = 0; layer < num_layers; ++layer) {
            const auto& edges = data.upper_layers[layer];
            uint32_t num_edges = static_cast<uint32_t>(edges.size());
            std::fwrite(&num_edges, sizeof(uint32_t), 1, f);

            for (const auto& edge : edges) {
                std::fwrite(&edge.node, sizeof(NodeId), 1, f);
                uint32_t num_neighbors = static_cast<uint32_t>(edge.neighbors.size());
                std::fwrite(&num_neighbors, sizeof(uint32_t), 1, f);
                if (num_neighbors > 0) {
                    std::fwrite(edge.neighbors.data(), sizeof(NodeId), num_neighbors, f);
                }
            }
        }
    }

    static void read_hnsw_layers(FILE* f, HNSWLayerSnapshot& data, uint64_t num_nodes) {
        if (std::fread(&data.max_level, sizeof(int), 1, f) != 1)
            throw std::runtime_error("Failed to read HNSW max_level");
        if (std::fread(&data.entry_point, sizeof(NodeId), 1, f) != 1)
            throw std::runtime_error("Failed to read HNSW entry_point");
        if (std::fread(&data.upper_tau, sizeof(float), 1, f) != 1)
            throw std::runtime_error("Failed to read HNSW upper_tau");

        uint64_t levels_size;
        if (std::fread(&levels_size, sizeof(uint64_t), 1, f) != 1)
            throw std::runtime_error("Failed to read node_levels size");
        data.node_levels.resize(levels_size);
        if (levels_size > 0) {
            if (std::fread(data.node_levels.data(), sizeof(int), levels_size, f) != levels_size)
                throw std::runtime_error("Failed to read node_levels");
        }

        uint32_t num_layers;
        if (std::fread(&num_layers, sizeof(uint32_t), 1, f) != 1)
            throw std::runtime_error("Failed to read num_layers");
        data.upper_layers.resize(num_layers);

        for (uint32_t layer = 0; layer < num_layers; ++layer) {
            uint32_t num_edges;
            if (std::fread(&num_edges, sizeof(uint32_t), 1, f) != 1)
                throw std::runtime_error("Failed to read num_edges");
            data.upper_layers[layer].resize(num_edges);

            for (uint32_t e = 0; e < num_edges; ++e) {
                auto& edge = data.upper_layers[layer][e];
                if (std::fread(&edge.node, sizeof(NodeId), 1, f) != 1)
                    throw std::runtime_error("Failed to read edge node");
                uint32_t num_neighbors;
                if (std::fread(&num_neighbors, sizeof(uint32_t), 1, f) != 1)
                    throw std::runtime_error("Failed to read num_neighbors");
                edge.neighbors.resize(num_neighbors);
                if (num_neighbors > 0) {
                    if (std::fread(edge.neighbors.data(), sizeof(NodeId), num_neighbors, f) != num_neighbors)
                        throw std::runtime_error("Failed to read edge neighbors");
                }
            }
        }
    }
};

}  // namespace cphnsw
