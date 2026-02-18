#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cphnsw/api/hnsw_index.hpp>
#include <cphnsw/core/adaptive_defaults.hpp>
#include <cphnsw/core/constants.hpp>
#include <cphnsw/core/util.hpp>

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

namespace py = pybind11;
using namespace cphnsw;

class PyIndexBase {
public:
    virtual ~PyIndexBase() = default;

    virtual void build(const float* vecs, size_t n) = 0;
    virtual void finalize() = 0;

    virtual size_t size() const = 0;
    virtual size_t dim() const = 0;
    virtual bool is_finalized() const = 0;

    virtual std::vector<SearchResult>
    search_raw(const float* query, const SearchRequest& request) const = 0;
};

template <size_t D, size_t BitWidth>
class PyIndexWrapper : public PyIndexBase {
public:
    using IndexType = Index<D, 32, BitWidth>;

    explicit PyIndexWrapper(size_t dim) {
        IndexParams params;
        params.dim = dim;
        index_ = std::make_unique<IndexType>(params);
    }

    void build(const float* vecs, size_t n) override {
        index_->build(vecs, n);
    }

    void finalize() override {
        index_->finalize();
    }

    size_t size() const override { return index_->size(); }
    size_t dim() const override { return index_->dim(); }
    bool is_finalized() const override { return index_->is_finalized(); }

    std::vector<SearchResult>
    search_raw(const float* query, const SearchRequest& request) const override {
        return index_->search(query, request);
    }

private:
    std::unique_ptr<IndexType> index_;
};

template <size_t BitWidth>
static std::unique_ptr<PyIndexBase> create_index_with_bits(size_t dim) {
    size_t pd = next_power_of_two(dim);

#define CASE_DIM(DIM) \
    case DIM: return std::make_unique<PyIndexWrapper<DIM, BitWidth>>(dim);

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

static std::unique_ptr<PyIndexBase> create_index(size_t dim, size_t bits) {
    switch (bits) {
        case 1: return create_index_with_bits<1>(dim);
        case 2: return create_index_with_bits<2>(dim);
        case 4: return create_index_with_bits<4>(dim);
        default:
            throw std::invalid_argument(
                "Unsupported bits=" + std::to_string(bits) +
                ". Supported: 1, 2, 4.");
    }
}

static SearchRequest make_search_request(size_t k, float recall_target) {
    SearchRequest req;
    req.k = k;
    req.target_recall = recall_target;
    return req;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "Calibration-Parameterless HNSW (CP-HNSW)";

    py::class_<PyIndexBase>(m, "CPIndex")
        .def(py::init([](size_t dim, size_t bits) {
                return create_index(dim, bits);
            }),
            py::arg("dim"),
            py::arg("bits") = 1)

        .def("build", [](PyIndexBase& self,
                          py::array_t<float, py::array::c_style | py::array::forcecast> vectors) {
            auto vbuf = vectors.request();
            if (vbuf.ndim != 2 || static_cast<size_t>(vbuf.shape[1]) != self.dim()) {
                throw std::invalid_argument("vectors must be a (n, dim) float32 array");
            }

            const float* vec_ptr = static_cast<const float*>(vbuf.ptr);
            size_t n = static_cast<size_t>(vbuf.shape[0]);

            py::gil_scoped_release release;
            self.build(vec_ptr, n);
        },
        py::arg("vectors"),
        "Build/rebuild the index from vectors.")

        .def("finalize", [](PyIndexBase& self) {
            py::gil_scoped_release release;
            self.finalize();
        }, "Finalize the index and fit calibration statistics.")

        .def("search", [](const PyIndexBase& self,
                          py::array_t<float, py::array::c_style | py::array::forcecast> query,
                          size_t k,
                          float recall_target) {
            auto buf = query.request();
            if (buf.ndim != 1 || static_cast<size_t>(buf.shape[0]) != self.dim()) {
                throw std::invalid_argument("query must be 1D and match index dimension");
            }

            const float* ptr = static_cast<const float*>(buf.ptr);
            SearchRequest request = make_search_request(k, recall_target);

            std::vector<SearchResult> results;
            {
                py::gil_scoped_release release;
                results = self.search_raw(ptr, request);
            }

            const size_t n = results.size();
            py::array_t<uint32_t> ids(n);
            py::array_t<float> distances(n);
            auto* ids_ptr = ids.mutable_data();
            auto* dist_ptr = distances.mutable_data();
            for (size_t i = 0; i < n; ++i) {
                ids_ptr[i] = results[i].id;
                dist_ptr[i] = results[i].distance;
            }
            return std::make_pair(ids, distances);
        },
        py::arg("query"),
        py::arg("k") = constants::kDefaultK,
        py::arg("recall_target") = constants::kDefaultRecall,
        "Search for nearest neighbors.")

        .def("search_batch", [](const PyIndexBase& self,
                                py::array_t<float, py::array::c_style | py::array::forcecast> queries,
                                size_t k,
                                float recall_target) {
            auto buf = queries.request();
            if (buf.ndim != 2 || static_cast<size_t>(buf.shape[1]) != self.dim()) {
                throw std::invalid_argument("queries must be a (n, dim) array");
            }

            const size_t n = static_cast<size_t>(buf.shape[0]);
            const float* ptr = static_cast<const float*>(buf.ptr);
            const size_t dim = self.dim();

            SearchRequest request = make_search_request(k, recall_target);

            py::array_t<int64_t> ids({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(k)});
            py::array_t<float> distances({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(k)});
            auto* ids_ptr = ids.mutable_data();
            auto* dist_ptr = distances.mutable_data();

            {
                py::gil_scoped_release release;
                const int actual_threads = omp_get_max_threads();
                const size_t omp_chunk = adaptive_defaults::omp_chunk_size(
                    n, static_cast<size_t>(actual_threads));

#pragma omp parallel for schedule(dynamic, omp_chunk) num_threads(actual_threads)
                for (size_t i = 0; i < n; ++i) {
                    auto results = self.search_raw(ptr + i * dim, request);
                    size_t j = 0;
                    for (; j < k && j < results.size(); ++j) {
                        ids_ptr[i * k + j] = static_cast<int64_t>(results[j].id);
                        dist_ptr[i * k + j] = results[j].distance;
                    }
                    for (; j < k; ++j) {
                        ids_ptr[i * k + j] = -1;
                        dist_ptr[i * k + j] = std::numeric_limits<float>::max();
                    }
                }
            }

            return std::make_pair(ids, distances);
        },
        py::arg("queries"),
        py::arg("k") = constants::kDefaultK,
        py::arg("recall_target") = constants::kDefaultRecall,
        "Batch search for nearest neighbors.")

        .def_property_readonly("size", &PyIndexBase::size,
                               "Total indexed nodes.")
        .def_property_readonly("dim", &PyIndexBase::dim,
                               "Original input dimensionality.")
        .def_property_readonly("is_finalized", &PyIndexBase::is_finalized,
                               "Whether finalize() has been run.");
}
