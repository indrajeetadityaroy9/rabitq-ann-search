#pragma once

#include "../graph/rabitq_graph.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace cphnsw {

// Binary format:
// Header: [magic(8)][version(4)][D(4)][R(4)][BitWidth(4)][n_nodes(8)][entry_point(4)][dim(4)]
// Centroid: [dim * float]
// Body: For each node: [code][neighbor_block][vector]
//   - Written as raw bytes of VertexData<D,R,BitWidth>

struct SerializationHeader {
    char magic[8] = {'R','B','Q','G','R','P','H', '\0'};
    uint32_t version = 1;
    uint32_t dims;
    uint32_t degree;
    uint32_t bit_width;
    uint64_t num_nodes;
    uint32_t entry_point;
    uint32_t original_dim;
};

template <size_t D, size_t R = 32, size_t BitWidth = 1>
class IndexSerializer {
public:
    using Graph = RaBitQGraph<D, R, BitWidth>;

    static void save(const std::string& path, const Graph& graph) {
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

        // Write each vertex's data (code + neighbors + vector)
        for (size_t i = 0; i < graph.size(); ++i) {
            NodeId id = static_cast<NodeId>(i);
            const auto& code = graph.get_code(id);
            std::fwrite(&code, sizeof(code), 1, f);

            const auto& nb = graph.get_neighbors(id);
            std::fwrite(&nb, sizeof(nb), 1, f);

            const float* vec = graph.get_vector(id);
            std::fwrite(vec, sizeof(float), D, f);
        }

        std::fclose(f);
    }

    static Graph load(const std::string& path) {
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

        using VertexDataType = typename Graph::VertexDataType;
        using CodeType = typename VertexDataType::CodeType;
        using NeighborBlockType = typename VertexDataType::NeighborBlockType;

        // First pass: add all nodes with codes and vectors
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
            // Restore neighbor block
            graph.get_neighbors(id) = nb;
        }

        graph.set_entry_point(static_cast<NodeId>(header.entry_point));
        std::fclose(f);
        return graph;
    }
};

}  // namespace cphnsw
