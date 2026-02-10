#pragma once

#include "../core/memory.hpp"
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace cphnsw {

// Implements a full Dense Random Orthogonal Rotation.
// Slower encoding O(D^2) than SRHT O(D log D), but provides 
// better theoretical concentration guarantees (no structural artifacts).
// Can be extended to support Learned Rotations (e.g., ITQ).
class DenseRotation {
public:
    DenseRotation(size_t dim, uint64_t seed)
        : dim_(dim)
        , padded_dim_(dim) // Dense rotation supports arbitrary dim, but we keep interface consistent
    {
        // Align padded_dim to 4/8/16 for SIMD if needed, but for now exact D
        if (dim_ % 4 != 0) {
            padded_dim_ = (dim_ + 3) / 4 * 4;
        }
        
        generate_orthogonal_matrix(seed);
    }

    void apply(float* x) const {
        // In-place apply is tricky for dense matrix without buffer.
        // We assume x has size padded_dim_
        
        // We need a temp buffer. Since this is called from encode_impl which
        // uses thread_local buffers, we might need to change the interface 
        // or allocate here. 
        // However, the RaBitQEncoder calls `apply_copy`.
        // If `apply` is called directly, we must be careful.
        
        std::vector<float> tmp(padded_dim_);
        apply_copy(x, tmp.data());
        std::memcpy(x, tmp.data(), padded_dim_ * sizeof(float));
    }

    void apply_copy(const float* input, float* output) const {
        // y = R * x
        // matrix_ is row-major (D x D)
        
        std::memset(output, 0, padded_dim_ * sizeof(float));
        
        // Simple O(D^2) multiplication. 
        // TODO: Optimize with BLAS or Tiling if D is large.
        // For D=128, this is small enough.
        
        for (size_t i = 0; i < dim_; ++i) {
            float sum = 0.0f;
            const float* row = matrix_.data() + i * dim_;
            
            // Vectorize this inner loop
            #pragma omp simd reduction(+:sum)
            for (size_t j = 0; j < dim_; ++j) {
                sum += row[j] * input[j];
            }
            output[i] = sum;
        }
    }

    size_t original_dim() const { return dim_; }
    size_t padded_dim() const { return padded_dim_; }
    
    // Support for learning/updating the matrix
    void set_matrix(const std::vector<float>& new_mat) {
        if (new_mat.size() != dim_ * dim_) {
            throw std::invalid_argument("Matrix size mismatch");
        }
        matrix_ = new_mat;
    }

    // ITQ Learned Rotation (Gong et al., TPAMI 2013).
    // Alternating optimization: fix R and quantize, fix quantization and solve
    // Procrustes for optimal R. Converges in ~50 iterations.
    // data: n x dim row-major, assumed centered and unit-normalized.
    void learn_rotation(const float* data, size_t n, size_t max_iters = 50) {
        if (n < dim_ || dim_ == 0) return;

        std::vector<float> rotated(n * dim_);
        std::vector<float> signs(n * dim_);
        std::vector<float> M(dim_ * dim_);

        for (size_t iter = 0; iter < max_iters; ++iter) {
            // 1. Rotate all training vectors: Y[k] = R * X[k]
            #pragma omp parallel for schedule(static)
            for (size_t k = 0; k < n; ++k) {
                apply_copy(data + k * dim_, rotated.data() + k * dim_);
            }

            // 2. Quantize: B = sign(Y)
            for (size_t i = 0; i < n * dim_; ++i) {
                signs[i] = (rotated[i] >= 0.0f) ? 1.0f : -1.0f;
            }

            // 3. M = X^T * B (dim x dim), the Procrustes target
            std::fill(M.begin(), M.end(), 0.0f);
            #pragma omp parallel
            {
                std::vector<float> M_local(dim_ * dim_, 0.0f);
                #pragma omp for schedule(static)
                for (size_t k = 0; k < n; ++k) {
                    const float* x = data + k * dim_;
                    const float* b = signs.data() + k * dim_;
                    for (size_t i = 0; i < dim_; ++i) {
                        float xi = x[i];
                        #pragma omp simd
                        for (size_t j = 0; j < dim_; ++j) {
                            M_local[i * dim_ + j] += xi * b[j];
                        }
                    }
                }
                #pragma omp critical
                for (size_t i = 0; i < dim_ * dim_; ++i) M[i] += M_local[i];
            }

            // 4. R = polar_factor(M) via Newton iteration
            polar_decomposition(M.data());

            // 5. Update rotation matrix
            matrix_ = M;
        }
    }

private:
    size_t dim_;
    size_t padded_dim_;
    std::vector<float> matrix_; // D x D flattened

    // Newton iteration for polar decomposition: X_{k+1} = (X_k + X_k^{-T}) / 2.
    // Converges quadratically to the orthogonal polar factor of M.
    void polar_decomposition(float* M) const {
        // Scale for stability: M /= ||M||_F * sqrt(dim)
        float m_norm = 0.0f;
        for (size_t i = 0; i < dim_ * dim_; ++i) m_norm += M[i] * M[i];
        m_norm = std::sqrt(m_norm);
        if (m_norm < 1e-10f) return;
        float scale = std::sqrt(static_cast<float>(dim_)) / m_norm;
        for (size_t i = 0; i < dim_ * dim_; ++i) M[i] *= scale;

        std::vector<float> Mt(dim_ * dim_);
        std::vector<float> inv_t(dim_ * dim_);

        for (size_t polar_iter = 0; polar_iter < 20; ++polar_iter) {
            // Transpose M
            for (size_t i = 0; i < dim_; ++i)
                for (size_t j = 0; j < dim_; ++j)
                    Mt[i * dim_ + j] = M[j * dim_ + i];

            // Invert M^T
            if (!invert_matrix(Mt.data(), inv_t.data())) break;

            // M = (M + inv(M^T)) / 2
            float diff = 0.0f;
            for (size_t i = 0; i < dim_ * dim_; ++i) {
                float new_val = (M[i] + inv_t[i]) * 0.5f;
                diff += (new_val - M[i]) * (new_val - M[i]);
                M[i] = new_val;
            }
            if (diff < 1e-10f * static_cast<float>(dim_ * dim_)) break;
        }
    }

    // Gaussian elimination with partial pivoting: inv = A^{-1}
    bool invert_matrix(const float* A, float* inv) const {
        size_t n = dim_;
        std::vector<float> work(n * n);
        std::memcpy(work.data(), A, n * n * sizeof(float));

        for (size_t i = 0; i < n * n; ++i) inv[i] = 0.0f;
        for (size_t i = 0; i < n; ++i) inv[i * n + i] = 1.0f;

        for (size_t col = 0; col < n; ++col) {
            // Partial pivot
            size_t max_row = col;
            float max_val = std::abs(work[col * n + col]);
            for (size_t row = col + 1; row < n; ++row) {
                float val = std::abs(work[row * n + col]);
                if (val > max_val) { max_val = val; max_row = row; }
            }
            if (max_val < 1e-12f) return false;

            if (max_row != col) {
                for (size_t j = 0; j < n; ++j) {
                    std::swap(work[col * n + j], work[max_row * n + j]);
                    std::swap(inv[col * n + j], inv[max_row * n + j]);
                }
            }

            float inv_pivot = 1.0f / work[col * n + col];
            for (size_t j = 0; j < n; ++j) {
                work[col * n + j] *= inv_pivot;
                inv[col * n + j] *= inv_pivot;
            }

            for (size_t row = 0; row < n; ++row) {
                if (row == col) continue;
                float factor = work[row * n + col];
                for (size_t j = 0; j < n; ++j) {
                    work[row * n + j] -= factor * work[col * n + j];
                    inv[row * n + j] -= factor * inv[col * n + j];
                }
            }
        }
        return true;
    }

    void generate_orthogonal_matrix(uint64_t seed) {
        matrix_.resize(dim_ * dim_);
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        // 1. Fill with Gaussian noise
        for (size_t i = 0; i < matrix_.size(); ++i) {
            matrix_[i] = dist(rng);
        }

        // 2. Gram-Schmidt Orthogonalization (Modified GS for stability)
        // Treat rows as vectors to be orthogonalized.
        
        for (size_t i = 0; i < dim_; ++i) {
            // Norm of row i
            float* row_i = matrix_.data() + i * dim_;
            
            float norm_sq = 0.0f;
            for (size_t k = 0; k < dim_; ++k) norm_sq += row_i[k] * row_i[k];
            float norm = std::sqrt(norm_sq);
            
            // Normalize
            float inv_norm = 1.0f / (norm + 1e-9f);
            for (size_t k = 0; k < dim_; ++k) row_i[k] *= inv_norm;
            
            // Subtract projection from subsequent rows
            for (size_t j = i + 1; j < dim_; ++j) {
                float* row_j = matrix_.data() + j * dim_;
                
                float dot = 0.0f;
                for (size_t k = 0; k < dim_; ++k) dot += row_i[k] * row_j[k];
                
                for (size_t k = 0; k < dim_; ++k) row_j[k] -= dot * row_i[k];
            }
        }
    }
};

}  // namespace cphnsw