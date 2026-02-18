#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cphnsw/api/hnsw_index.hpp>
#include <cphnsw/core/adaptive_defaults.hpp>
#include <cphnsw/core/constants.hpp>
#include <cphnsw/core/util.hpp>
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
    virtual void finalize() = 0;
    virtual size_t size() const = 0;
    virtual size_t dim() const = 0;
    virtual bool is_finalized() const = 0;

    virtual std::vector<SearchResult>
    search_raw(const float* query, const SearchParams& params) const = 0;

    virtual void save(const std::string& path) const = 0;

    virtual py::dict calibration_info() const = 0;
};

template <size_t D, size_t BitWidth>
class PyIndexWrapper : public PyIndexBase {
public:
    using IndexType = Index<D, 32, BitWidth>;
    using Graph = typename IndexType::Graph;

    explicit PyIndexWrapper(size_t dim) {
        IndexParams params;
        params.dim = dim;
        index_ = std::make_unique<IndexType>(params);
    }

    PyIndexWrapper(size_t dim,
                   Graph&& graph, const HNSWLayerSnapshot& layer_data,
                   const CalibrationSnapshot& cal = CalibrationSnapshot(),
                   uint64_t rotation_seed = constants::kDefaultRotationSeed) {
        IndexParams params;
        params.dim = dim;
        index_ = std::make_unique<IndexType>(params, std::move(graph), layer_data, cal, rotation_seed);
    }

    void add_batch(const float* vecs, size_t n) override { index_->add_batch(vecs, n); }
    void finalize() override { index_->finalize(); }
    size_t size() const override { return index_->size(); }
    size_t dim() const override { return index_->dim(); }
    bool is_finalized() const override { return index_->is_finalized(); }

    std::vector<SearchResult>
    search_raw(const float* query, const SearchParams& params) const override {
        return index_->search(query, params);
    }

    void save(const std::string& path) const override {
        auto snap = index_->get_layer_snapshot();
        auto cal = index_->get_calibration_snapshot();
        IndexSerializer<D, 32, BitWidth>::save(path, index_->graph(), snap, cal,
                                                index_->rotation_seed());
    }

    py::dict calibration_info() const override {
        auto cal = index_->get_calibration_snapshot();
        py::dict info;
        info["affine_a"] = cal.affine_a;
        info["affine_b"] = cal.affine_b;
        info["ip_qo_floor"] = cal.ip_qo_floor;
        info["resid_sigma"] = cal.resid_sigma;
        info["resid_q99_dot"] = cal.resid_q99_dot;
        info["median_nn_dist_sq"] = cal.median_nn_dist_sq;
        info["calibration_corr"] = cal.calibration_corr;
        info["calibrated"] = (cal.flags & 1u) != 0;

        py::dict evt;
        evt["fitted"] = cal.evt.fitted;
        evt["u"] = cal.evt.u;
        evt["p_u"] = cal.evt.p_u;
        evt["xi"] = cal.evt.xi;
        evt["beta"] = cal.evt.beta;
        evt["nop_p95"] = cal.evt.nop_p95;
        evt["n_resid"] = cal.evt.n_resid;
        evt["n_tail"] = cal.evt.n_tail;
        info["evt"] = evt;
        info["rotation_seed"] = index_->rotation_seed();

        return info;
    }

private:
    std::unique_ptr<IndexType> index_;
};


template <size_t BitWidth>
static std::unique_ptr<PyIndexBase> create_index_with_bits(
    size_t dim)
{
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

static std::unique_ptr<PyIndexBase> create_index(
    size_t dim, size_t bits)
{
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


template <size_t BitWidth>
static std::unique_ptr<PyIndexBase> load_index_with_bits(
    const std::string& path, size_t original_dim, size_t pd)
{
    #define LOAD_CASE_DIM(DIM) \
        case DIM: { \
            auto result = IndexSerializer<DIM, 32, BitWidth>::load(path); \
            return std::make_unique<PyIndexWrapper<DIM, BitWidth>>( \
                original_dim, std::move(result.graph), result.hnsw_data, \
                result.calibration, result.rotation_seed); \
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
    const std::string& path)
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
    if (header.version != SERIALIZATION_VERSION) {
        throw std::runtime_error("Unsupported serialization version: " +
            std::to_string(header.version) + " (expected " +
            std::to_string(SERIALIZATION_VERSION) + ")");
    }

    size_t original_dim = header.original_dim;
    size_t pd = header.dims;
    size_t bits = header.bit_width;

    switch (bits) {
        case 1: return load_index_with_bits<1>(path, original_dim, pd);
        case 2: return load_index_with_bits<2>(path, original_dim, pd);
        case 4: return load_index_with_bits<4>(path, original_dim, pd);
        default:
            throw std::invalid_argument(
                "Unsupported bit_width=" + std::to_string(bits) +
                " in saved file. Supported: 1, 2, 4.");
    }
}


PYBIND11_MODULE(_core, m) {
    m.doc() = "Configuration-Parameterless HNSW (CP-HNSW): Zero-tuning RaBitQ approximate nearest neighbor search";

    py::class_<PyIndexBase>(m, "Index")
        .def(py::init([](size_t dim, size_t bits) {
            return create_index(dim, bits);
        }),
            py::arg("dim"),
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

        .def("finalize", [](PyIndexBase& self) {
            py::gil_scoped_release release;
            self.finalize();
        },
           "Finalize the index and run mandatory calibration/EVT fitting.")

        .def("search", [](const PyIndexBase& self,
                          py::array_t<float, py::array::c_style | py::array::forcecast> query,
                          size_t k, float recall_target) {
            auto buf = query.request();
            if (buf.ndim != 1 || static_cast<size_t>(buf.shape[0]) != self.dim())
                throw std::invalid_argument("Query must be 1D array matching index dimension");
            const float* ptr = static_cast<const float*>(buf.ptr);

            SearchParams params;
            params.k = k;
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
        }, py::arg("query"), py::arg("k") = constants::kDefaultK, py::arg("recall_target") = constants::kDefaultRecall,
           "Search for k nearest neighbors. Returns (ids, distances) arrays.\n"
           "recall_target controls the quality/speed tradeoff (default 0.95).")

        .def("search_batch", [](const PyIndexBase& self,
                               py::array_t<float, py::array::c_style | py::array::forcecast> queries,
                               size_t k, float recall_target) {
            auto buf = queries.request();
            if (buf.ndim != 2 || static_cast<size_t>(buf.shape[1]) != self.dim())
                throw std::invalid_argument("queries must be (n, dim) array");

            size_t n = static_cast<size_t>(buf.shape[0]);
            const float* ptr = static_cast<const float*>(buf.ptr);
            size_t dim = self.dim();

            SearchParams params;
            params.k = k;
            params.recall_target = recall_target;

            py::array_t<int64_t> ids({n, k});
            py::array_t<float> distances({n, k});
            auto ids_ptr = ids.mutable_data();
            auto dist_ptr = distances.mutable_data();

            {
                py::gil_scoped_release release;
                int actual_threads = omp_get_max_threads();
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
        }, py::arg("queries"), py::arg("k") = constants::kDefaultK,
           py::arg("recall_target") = constants::kDefaultRecall,
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
           "Save the index graph to a binary file.")

        .def_property_readonly("calibration_info", &PyIndexBase::calibration_info,
                               "Calibration state: {affine_a, affine_b, ip_qo_floor, resid_sigma, "
                               "resid_q99_dot, median_nn_dist_sq, calibration_corr, calibrated}")
        ;

    m.def("load", [](const std::string& path) {
        return load_index(path);
    },
        py::arg("path"),
        "Load an index from a binary file previously saved with index.save().");
}
