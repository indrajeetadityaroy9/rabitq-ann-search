#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cphnsw/api/rabitq_index.hpp>

#include <memory>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using namespace cphnsw;

// ============================================================================
// Type-erased index wrapper (RaBitQIndex is templated on D)
// ============================================================================

class PyIndexBase {
public:
    virtual ~PyIndexBase() = default;

    virtual void add_batch(const float* vecs, size_t n) = 0;
    virtual void finalize(bool verbose) = 0;
    virtual size_t size() const = 0;
    virtual size_t dim() const = 0;
    virtual bool is_finalized() const = 0;

    virtual std::vector<SearchResult>
    search_raw(const float* query, size_t k, size_t ef) const = 0;
};

template <size_t D>
class PyIndex : public PyIndexBase {
public:
    PyIndex(size_t dim, size_t M, size_t ef_construction, uint64_t seed) {
        IndexParams params;
        params.dim = dim;
        params.M = M;
        params.ef_construction = ef_construction;
        params.seed = seed;
        index_ = std::make_unique<RaBitQIndex<D, 32>>(params);
    }

    void add_batch(const float* vecs, size_t n) override {
        index_->add_batch(vecs, n);
    }

    void finalize(bool verbose) override {
        index_->finalize(0, verbose);
    }

    size_t size() const override { return index_->size(); }
    size_t dim() const override { return index_->dim(); }
    bool is_finalized() const override { return index_->is_finalized(); }

    std::vector<SearchResult>
    search_raw(const float* query, size_t k, size_t ef) const override {
        return index_->search(query, SearchParams().set_k(k).set_ef(ef));
    }

private:
    std::unique_ptr<RaBitQIndex<D, 32>> index_;
};

// ============================================================================
// Factory: select template instantiation based on dimension
// ============================================================================

static size_t padded_dim(size_t dim) {
    size_t p = 1;
    while (p < dim) p *= 2;
    return p;
}

static std::unique_ptr<PyIndexBase> create_index(
    size_t dim, size_t M, size_t ef_construction, uint64_t seed)
{
    size_t pd = padded_dim(dim);
    switch (pd) {
        case 128:  return std::make_unique<PyIndex<128>>(dim, M, ef_construction, seed);
        case 256:  return std::make_unique<PyIndex<256>>(dim, M, ef_construction, seed);
        case 512:  return std::make_unique<PyIndex<512>>(dim, M, ef_construction, seed);
        case 1024: return std::make_unique<PyIndex<1024>>(dim, M, ef_construction, seed);
        default:
            throw std::invalid_argument(
                "Unsupported dimension " + std::to_string(dim) +
                " (padded to " + std::to_string(pd) +
                "). Supported padded dims: 128, 256, 512, 1024.");
    }
}

// ============================================================================
// Python module
// ============================================================================

PYBIND11_MODULE(_core, m) {
    m.doc() = "CP-HNSW: RaBitQ + SymphonyQG approximate nearest neighbor search";

    py::class_<PyIndexBase>(m, "Index")
        .def(py::init([](size_t dim, size_t M, size_t ef_construction, uint64_t seed) {
            return create_index(dim, M, ef_construction, seed);
        }),
            py::arg("dim"),
            py::arg("M") = 32,
            py::arg("ef_construction") = 200,
            py::arg("seed") = 42)

        .def("add", [](PyIndexBase& self, py::array_t<float, py::array::c_style | py::array::forcecast> vectors) {
            auto buf = vectors.request();
            if (buf.ndim == 1) {
                // Single vector
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

        .def("finalize", [](PyIndexBase& self, bool verbose) {
            py::gil_scoped_release release;
            self.finalize(verbose);
        }, py::arg("verbose") = false,
           "Finalize the index (graph refinement + medoid selection).")

        .def("search", [](const PyIndexBase& self,
                          py::array_t<float, py::array::c_style | py::array::forcecast> query,
                          size_t k, size_t ef) {
            auto buf = query.request();
            if (buf.ndim != 1 || static_cast<size_t>(buf.shape[0]) != self.dim())
                throw std::invalid_argument("Query must be 1D array matching index dimension");
            const float* ptr = static_cast<const float*>(buf.ptr);

            // Run C++ search without GIL
            std::vector<SearchResult> results;
            {
                py::gil_scoped_release release;
                results = self.search_raw(ptr, k, ef);
            }

            // Create numpy arrays with GIL held
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
        }, py::arg("query"), py::arg("k") = 10, py::arg("ef") = 100,
           "Search for k nearest neighbors. Returns (ids, distances) arrays.")

        .def_property_readonly("size", &PyIndexBase::size,
                               "Number of vectors in the index.")
        .def_property_readonly("dim", &PyIndexBase::dim,
                               "Vector dimension.")
        .def_property_readonly("is_finalized", &PyIndexBase::is_finalized,
                               "Whether finalize() has been called.");
}
