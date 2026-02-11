#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cphnsw/api/rabitq_index.hpp>
#include <cphnsw/api/hnsw_index.hpp>
#include <cphnsw/core/adaptive_defaults.hpp>
#include <cphnsw/io/serialization.hpp>

#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#include <omp.h>

namespace py = pybind11;
using namespace cphnsw;

// ============================================================================
// Type-erased index wrapper (RaBitQIndex is templated on D, R, BitWidth)
// ============================================================================

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

template <size_t D, size_t BitWidth = 1>
class PyIndex : public PyIndexBase {
public:
    PyIndex(size_t dim, size_t ef_construction, uint64_t seed) {
        IndexParams params;
        params.dim = dim;
        params.ef_construction = ef_construction;
        params.seed = seed;
        index_ = std::make_unique<RaBitQIndex<D, 32, BitWidth>>(params);
    }

    void add_batch(const float* vecs, size_t n) override {
        index_->add_batch(vecs, n);
    }

    void finalize(const BuildParams& params) override {
        index_->finalize(params);
    }

    size_t size() const override { return index_->size(); }
    size_t dim() const override { return index_->dim(); }
    bool is_finalized() const override { return index_->is_finalized(); }

    std::vector<SearchResult>
    search_raw(const float* query, const SearchParams& params) const override {
        return index_->search(query, params);
    }

    void save(const std::string& path) const override {
        IndexSerializer<D, 32, BitWidth>::save(path, index_->graph());
    }

private:
    std::unique_ptr<RaBitQIndex<D, 32, BitWidth>> index_;
};

// ============================================================================
// Type-erased HNSW index wrapper
// ============================================================================

template <size_t D, size_t BitWidth = 1>
class PyHNSWIndex : public PyIndexBase {
public:
    PyHNSWIndex(size_t dim, size_t ef_construction, uint64_t seed) {
        IndexParams params;
        params.dim = dim;
        params.ef_construction = ef_construction;
        params.seed = seed;
        index_ = std::make_unique<HNSWIndex<D, 32, BitWidth>>(params);
    }

    void add_batch(const float* vecs, size_t n) override {
        index_->add_batch(vecs, n);
    }

    void finalize(const BuildParams& params) override {
        index_->finalize(params);
    }

    size_t size() const override { return index_->size(); }
    size_t dim() const override { return index_->dim(); }
    bool is_finalized() const override { return index_->is_finalized(); }

    std::vector<SearchResult>
    search_raw(const float* query, const SearchParams& params) const override {
        return index_->search(query, params);
    }

    void save(const std::string& path) const override {
        IndexSerializer<D, 32, BitWidth>::save(path, index_->graph());
    }

private:
    std::unique_ptr<HNSWIndex<D, 32, BitWidth>> index_;
};

// ============================================================================
// Factory: select template instantiation based on dimension and bits
// ============================================================================

static size_t padded_dim(size_t dim) {
    size_t p = 1;
    while (p < dim) p *= 2;
    return p;
}

template <size_t BitWidth>
static std::unique_ptr<PyIndexBase> create_index_with_bits(
    size_t dim, size_t ef_construction, uint64_t seed)
{
    size_t pd = padded_dim(dim);
    switch (pd) {
        case 128:  return std::make_unique<PyIndex<128, BitWidth>>(dim, ef_construction, seed);
        case 256:  return std::make_unique<PyIndex<256, BitWidth>>(dim, ef_construction, seed);
        case 512:  return std::make_unique<PyIndex<512, BitWidth>>(dim, ef_construction, seed);
        case 1024: return std::make_unique<PyIndex<1024, BitWidth>>(dim, ef_construction, seed);
        default:
            throw std::invalid_argument(
                "Unsupported dimension " + std::to_string(dim) +
                " (padded to " + std::to_string(pd) +
                "). Supported padded dims: 128, 256, 512, 1024.");
    }
}

static std::unique_ptr<PyIndexBase> create_index(
    size_t dim, size_t ef_construction, uint64_t seed, size_t bits)
{
    switch (bits) {
        case 1: return create_index_with_bits<1>(dim, ef_construction, seed);
        case 2: return create_index_with_bits<2>(dim, ef_construction, seed);
        case 4: return create_index_with_bits<4>(dim, ef_construction, seed);
        default:
            throw std::invalid_argument(
                "Unsupported bits=" + std::to_string(bits) +
                ". Supported: 1, 2, 4.");
    }
}

// ============================================================================
// HNSW factory
// ============================================================================

template <size_t BitWidth>
static std::unique_ptr<PyIndexBase> create_hnsw_with_bits(
    size_t dim, size_t ef_construction, uint64_t seed)
{
    size_t pd = padded_dim(dim);
    switch (pd) {
        case 128:  return std::make_unique<PyHNSWIndex<128, BitWidth>>(dim, ef_construction, seed);
        case 256:  return std::make_unique<PyHNSWIndex<256, BitWidth>>(dim, ef_construction, seed);
        case 512:  return std::make_unique<PyHNSWIndex<512, BitWidth>>(dim, ef_construction, seed);
        case 1024: return std::make_unique<PyHNSWIndex<1024, BitWidth>>(dim, ef_construction, seed);
        default:
            throw std::invalid_argument(
                "Unsupported dimension " + std::to_string(dim) +
                " (padded to " + std::to_string(pd) +
                "). Supported padded dims: 128, 256, 512, 1024.");
    }
}

static std::unique_ptr<PyIndexBase> create_hnsw(
    size_t dim, size_t ef_construction, uint64_t seed, size_t bits)
{
    switch (bits) {
        case 1: return create_hnsw_with_bits<1>(dim, ef_construction, seed);
        case 2: return create_hnsw_with_bits<2>(dim, ef_construction, seed);
        case 4: return create_hnsw_with_bits<4>(dim, ef_construction, seed);
        default:
            throw std::invalid_argument(
                "Unsupported bits=" + std::to_string(bits) +
                ". Supported: 1, 2, 4.");
    }
}

// ============================================================================
// Python module
// ============================================================================

PYBIND11_MODULE(_core, m) {
    m.doc() = "CP-HNSW: Zero-tuning RaBitQ approximate nearest neighbor search";

    py::class_<PyIndexBase>(m, "Index")
        .def(py::init([](size_t dim, size_t M, size_t ef_construction, uint64_t seed, size_t bits) {
            if (M != 32) {
                throw std::invalid_argument(
                    "M=" + std::to_string(M) +
                    " not supported. Graph degree is compiled with R=32; only M=32 is valid.");
            }
            return create_index(dim, ef_construction, seed, bits);
        }),
            py::arg("dim"),
            py::arg("M") = 32,
            py::arg("ef_construction") = 200,
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

        .def("finalize", [](PyIndexBase& self, bool verbose,
                            size_t ef_construction, size_t num_threads,
                            float error_tolerance, float error_epsilon) {
            BuildParams params;
            params.verbose = verbose;
            params.ef_construction = ef_construction;
            params.num_threads = num_threads;
            params.error_tolerance = error_tolerance;
            params.error_epsilon = error_epsilon;
            py::gil_scoped_release release;
            self.finalize(params);
        }, py::arg("verbose") = false,
           py::arg("ef_construction") = 0,
           py::arg("num_threads") = 0,
           py::arg("error_tolerance") = -1.0f,
           py::arg("error_epsilon") = 0.0f,
           "Finalize the index (adaptive two-pass graph construction).\n"
           "All parameters auto-derive from data statistics by default.\n"
           "Set ef_construction>0 to override beam width.\n"
           "Set error_tolerance>=0 to override quantization error margin.\n"
           "Set error_epsilon>0 to override bound looseness.")

        .def("search", [](const PyIndexBase& self,
                          py::array_t<float, py::array::c_style | py::array::forcecast> query,
                          size_t k, size_t ef, float error_epsilon, float recall_target) {
            auto buf = query.request();
            if (buf.ndim != 1 || static_cast<size_t>(buf.shape[0]) != self.dim())
                throw std::invalid_argument("Query must be 1D array matching index dimension");
            const float* ptr = static_cast<const float*>(buf.ptr);

            SearchParams params;
            params.k = k;
            params.ef = ef;
            params.error_epsilon = error_epsilon;
            params.recall_target = recall_target;

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
        }, py::arg("query"), py::arg("k") = 10, py::arg("ef") = 0,
           py::arg("error_epsilon") = 0.0f, py::arg("recall_target") = 0.95f,
           "Search for k nearest neighbors. Returns (ids, distances) arrays.\n"
           "By default, ef and error_epsilon auto-derive from recall_target.\n"
           "Set ef>0 or error_epsilon>0 to override.")

        .def("search_batch", [](const PyIndexBase& self,
                               py::array_t<float, py::array::c_style | py::array::forcecast> queries,
                               size_t k, size_t ef, int n_threads,
                               float error_epsilon, float recall_target) {
            auto buf = queries.request();
            if (buf.ndim != 2 || static_cast<size_t>(buf.shape[1]) != self.dim())
                throw std::invalid_argument("queries must be (n, dim) array");

            size_t n = static_cast<size_t>(buf.shape[0]);
            const float* ptr = static_cast<const float*>(buf.ptr);
            size_t dim = self.dim();

            SearchParams params;
            params.k = k;
            params.ef = ef;
            params.error_epsilon = error_epsilon;
            params.recall_target = recall_target;

            py::array_t<int64_t> ids({n, k});
            py::array_t<float> distances({n, k});
            auto ids_ptr = ids.mutable_data();
            auto dist_ptr = distances.mutable_data();

            {
                py::gil_scoped_release release;
                int actual_threads = n_threads > 0 ? n_threads : omp_get_max_threads();
                #pragma omp parallel for schedule(dynamic, 16) num_threads(actual_threads)
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
        }, py::arg("queries"), py::arg("k") = 10, py::arg("ef") = 0,
           py::arg("n_threads") = 0, py::arg("error_epsilon") = 0.0f,
           py::arg("recall_target") = 0.95f,
           "Batch search for k nearest neighbors (OpenMP parallel). Returns (ids, distances) as (n,k) arrays.\n"
           "By default, ef and error_epsilon auto-derive from recall_target.")

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

    // HNSW multi-layer index — factory function returning same PyIndexBase interface
    m.def("HNSWIndex", [](size_t dim, size_t M, size_t ef_construction, uint64_t seed, size_t bits) {
        if (M != 32) {
            throw std::invalid_argument(
                "M=" + std::to_string(M) +
                " not supported. Graph degree is compiled with R=32; only M=32 is valid.");
        }
        return create_hnsw(dim, ef_construction, seed, bits);
    },
        py::arg("dim"),
        py::arg("M") = 32,
        py::arg("ef_construction") = 200,
        py::arg("seed") = 42,
        py::arg("bits") = 1,
        "Create an HNSW multi-layer index. Same interface as Index but uses "
        "hierarchical routing for faster search on large datasets.");
}
