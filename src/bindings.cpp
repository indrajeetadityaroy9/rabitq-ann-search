#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cphnsw/api/hnsw_index.hpp>
#include <cphnsw/core/adaptive_defaults.hpp>
#include <cphnsw/io/serialization.hpp>

#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#include <omp.h>

namespace py = pybind11;
using namespace cphnsw;


class PyIndexBase {
public:
    virtual ~PyIndexBase() = default;

    virtual void add_batch(const float* vecs, size_t n) = 0;
    virtual void finalize(const BuildParams& params) = 0;
    virtual size_t size() const = 0;
    virtual size_t dim() const = 0;
    virtual bool is_finalized() const = 0;

    virtual std::vector<SearchResult>
    search_raw(const float* query, const SearchParams& params) const = 0;

    virtual void save(const std::string& path) const = 0;
};

template <size_t D, size_t BitWidth>
class PyIndexWrapper : public PyIndexBase {
public:
    using IndexType = Index<D, 32, BitWidth>;
    using Graph = typename IndexType::Graph;

    PyIndexWrapper(size_t dim, uint64_t seed) {
        IndexParams params;
        params.dim = dim;
        params.seed = seed;
        index_ = std::make_unique<IndexType>(params);
        seed_ = seed;
    }

    PyIndexWrapper(size_t dim, uint64_t seed,
                   Graph&& graph, const HNSWLayerSnapshot& layer_data) {
        IndexParams params;
        params.dim = dim;
        params.seed = seed;
        index_ = std::make_unique<IndexType>(params, std::move(graph), layer_data);
        seed_ = seed;
    }

    void add_batch(const float* vecs, size_t n) override { index_->add_batch(vecs, n); }
    void finalize(const BuildParams& params) override { index_->finalize(params); }
    size_t size() const override { return index_->size(); }
    size_t dim() const override { return index_->dim(); }
    bool is_finalized() const override { return index_->is_finalized(); }

    std::vector<SearchResult>
    search_raw(const float* query, const SearchParams& params) const override {
        return index_->search(query, params);
    }

    void save(const std::string& path) const override {
        auto snap = index_->get_layer_snapshot();
        IndexSerializer<D, 32, BitWidth>::save(path, index_->graph(), snap);
    }

private:
    std::unique_ptr<IndexType> index_;
    uint64_t seed_;
};


static size_t padded_dim(size_t dim) {
    size_t p = 1;
    while (p < dim) p *= 2;
    return p;
}

template <size_t BitWidth>
static std::unique_ptr<PyIndexBase> create_index_with_bits(
    size_t dim, uint64_t seed)
{
    size_t pd = padded_dim(dim);

    #define CASE_DIM(DIM) \
        case DIM: return std::make_unique<PyIndexWrapper<DIM, BitWidth>>(dim, seed);

    switch (pd) {
        CASE_DIM(16)
        CASE_DIM(32)
        CASE_DIM(64)
        CASE_DIM(128)
        CASE_DIM(256)
        CASE_DIM(512)
        CASE_DIM(1024)
        CASE_DIM(2048)
        default:
            throw std::invalid_argument(
                "Unsupported dimension " + std::to_string(dim) +
                " (padded to " + std::to_string(pd) +
                "). Supported padded dims: 16, 32, 64, 128, 256, 512, 1024, 2048.");
    }

    #undef CASE_DIM
}

static std::unique_ptr<PyIndexBase> create_index(
    size_t dim, uint64_t seed, size_t bits)
{
    switch (bits) {
        case 1: return create_index_with_bits<1>(dim, seed);
        case 2: return create_index_with_bits<2>(dim, seed);
        case 4: return create_index_with_bits<4>(dim, seed);
        default:
            throw std::invalid_argument(
                "Unsupported bits=" + std::to_string(bits) +
                ". Supported: 1, 2, 4.");
    }
}


template <size_t BitWidth>
static std::unique_ptr<PyIndexBase> load_index_with_bits(
    const std::string& path, size_t original_dim, size_t pd, uint64_t seed)
{
    #define LOAD_CASE_DIM(DIM) \
        case DIM: { \
            auto result = IndexSerializer<DIM, 32, BitWidth>::load(path); \
            return std::make_unique<PyIndexWrapper<DIM, BitWidth>>( \
                original_dim, seed, std::move(result.graph), result.hnsw_data); \
        }

    switch (pd) {
        LOAD_CASE_DIM(16)
        LOAD_CASE_DIM(32)
        LOAD_CASE_DIM(64)
        LOAD_CASE_DIM(128)
        LOAD_CASE_DIM(256)
        LOAD_CASE_DIM(512)
        LOAD_CASE_DIM(1024)
        LOAD_CASE_DIM(2048)
        default:
            throw std::invalid_argument(
                "Unsupported padded dimension " + std::to_string(pd));
    }

    #undef LOAD_CASE_DIM
}

static std::unique_ptr<PyIndexBase> load_index(
    const std::string& path, uint64_t seed)
{
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("Cannot open file for reading: " + path);

    SerializationHeader header;
    if (std::fread(&header, sizeof(header), 1, f) != 1) {
        std::fclose(f);
        throw std::runtime_error("Failed to read header");
    }
    std::fclose(f);

    if (std::memcmp(header.magic, "RBQGRPH", 7) != 0) {
        throw std::runtime_error("Invalid file format");
    }

    size_t original_dim = header.original_dim;
    size_t pd = header.dims;
    size_t bits = header.bit_width;

    switch (bits) {
        case 1: return load_index_with_bits<1>(path, original_dim, pd, seed);
        case 2: return load_index_with_bits<2>(path, original_dim, pd, seed);
        case 4: return load_index_with_bits<4>(path, original_dim, pd, seed);
        default:
            throw std::invalid_argument(
                "Unsupported bit_width=" + std::to_string(bits) +
                " in saved file. Supported: 1, 2, 4.");
    }
}


PYBIND11_MODULE(_core, m) {
    m.doc() = "Configuration-Parameterless HNSW (CP-HNSW): Zero-tuning RaBitQ approximate nearest neighbor search";

    py::class_<PyIndexBase>(m, "Index")
        .def(py::init([](size_t dim, uint64_t seed, size_t bits) {
            return create_index(dim, seed, bits);
        }),
            py::arg("dim"),
            py::arg("seed") = 42,
            py::arg("bits") = 1)

        .def("add", [](PyIndexBase& self, py::array_t<float, py::array::c_style | py::array::forcecast> vectors) {
            auto buf = vectors.request();
            if (buf.ndim == 1) {
                if (static_cast<size_t>(buf.shape[0]) != self.dim())
                    throw std::invalid_argument("Vector dimension mismatch");
                std::vector<float> tmp(static_cast<float*>(buf.ptr),
                                       static_cast<float*>(buf.ptr) + buf.shape[0]);
                py::gil_scoped_release release;
                self.add_batch(tmp.data(), 1);
            } else if (buf.ndim == 2) {
                if (static_cast<size_t>(buf.shape[1]) != self.dim())
                    throw std::invalid_argument("Vector dimension mismatch");
                size_t n = static_cast<size_t>(buf.shape[0]);
                const float* ptr = static_cast<const float*>(buf.ptr);
                py::gil_scoped_release release;
                self.add_batch(ptr, n);
            } else {
                throw std::invalid_argument("Expected 1D or 2D array");
            }
        }, py::arg("vectors"),
           "Add vector(s) to the index. Accepts (dim,) or (n, dim) arrays.")

        .def("finalize", [](PyIndexBase& self, bool verbose, size_t num_threads) {
            BuildParams params;
            params.verbose = verbose;
            params.num_threads = num_threads;
            py::gil_scoped_release release;
            self.finalize(params);
        }, py::arg("verbose") = false,
           py::arg("num_threads") = 0,
           "Finalize the index. All construction parameters are derived automatically.")

        .def("search", [](const PyIndexBase& self,
                          py::array_t<float, py::array::c_style | py::array::forcecast> query,
                          size_t k, float recall_target, float gamma) {
            auto buf = query.request();
            if (buf.ndim != 1 || static_cast<size_t>(buf.shape[0]) != self.dim())
                throw std::invalid_argument("Query must be 1D array matching index dimension");
            const float* ptr = static_cast<const float*>(buf.ptr);

            SearchParams params;
            params.k = k;
            params.recall_target = recall_target;
            params.gamma_override = gamma;

            std::vector<SearchResult> results;
            {
                py::gil_scoped_release release;
                results = self.search_raw(ptr, params);
            }

            size_t n = results.size();
            py::array_t<uint32_t> ids(n);
            py::array_t<float> distances(n);
            auto ids_ptr = ids.mutable_data();
            auto dist_ptr = distances.mutable_data();
            for (size_t i = 0; i < n; ++i) {
                ids_ptr[i] = results[i].id;
                dist_ptr[i] = results[i].distance;
            }
            return std::make_pair(ids, distances);
        }, py::arg("query"), py::arg("k") = 10, py::arg("recall_target") = 0.95f,
           py::arg("gamma") = -1.0f,
           "Search for k nearest neighbors. Returns (ids, distances) arrays.\n"
           "recall_target controls the quality/speed tradeoff (default 0.95).\n"
           "gamma overrides the beam exploration budget (negative = derive from recall_target).")

        .def("search_batch", [](const PyIndexBase& self,
                               py::array_t<float, py::array::c_style | py::array::forcecast> queries,
                               size_t k, int n_threads, float recall_target, float gamma) {
            auto buf = queries.request();
            if (buf.ndim != 2 || static_cast<size_t>(buf.shape[1]) != self.dim())
                throw std::invalid_argument("queries must be (n, dim) array");

            size_t n = static_cast<size_t>(buf.shape[0]);
            const float* ptr = static_cast<const float*>(buf.ptr);
            size_t dim = self.dim();

            SearchParams params;
            params.k = k;
            params.recall_target = recall_target;
            params.gamma_override = gamma;

            py::array_t<int64_t> ids({n, k});
            py::array_t<float> distances({n, k});
            auto ids_ptr = ids.mutable_data();
            auto dist_ptr = distances.mutable_data();

            {
                py::gil_scoped_release release;
                int actual_threads = n_threads > 0 ? n_threads : omp_get_max_threads();
                size_t omp_chunk = adaptive_defaults::omp_chunk_size(n, static_cast<size_t>(actual_threads));
                #pragma omp parallel for schedule(dynamic, omp_chunk) num_threads(actual_threads)
                for (size_t i = 0; i < n; ++i) {
                    auto results = self.search_raw(ptr + i * dim, params);
                    for (size_t j = 0; j < k && j < results.size(); ++j) {
                        ids_ptr[i * k + j] = static_cast<int64_t>(results[j].id);
                        dist_ptr[i * k + j] = results[j].distance;
                    }
                    for (size_t j = results.size(); j < k; ++j) {
                        ids_ptr[i * k + j] = -1;
                        dist_ptr[i * k + j] = std::numeric_limits<float>::max();
                    }
                }
            }
            return std::make_pair(ids, distances);
        }, py::arg("queries"), py::arg("k") = 10,
           py::arg("n_threads") = 0, py::arg("recall_target") = 0.95f,
           py::arg("gamma") = -1.0f,
           "Batch search for k nearest neighbors (OpenMP parallel). Returns (ids, distances) as (n,k) arrays.")

        .def_property_readonly("size", &PyIndexBase::size,
                               "Number of vectors in the index.")
        .def_property_readonly("dim", &PyIndexBase::dim,
                               "Vector dimension.")
        .def_property_readonly("is_finalized", &PyIndexBase::is_finalized,
                               "Whether finalize() has been called.")

        .def("save", [](const PyIndexBase& self, const std::string& path) {
            self.save(path);
        }, py::arg("path"),
           "Save the index graph to a binary file.");

    m.def("load", [](const std::string& path, uint64_t seed) {
        return load_index(path, seed);
    },
        py::arg("path"),
        py::arg("seed") = 42,
        "Load an index from a binary file previously saved with index.save().");
}
