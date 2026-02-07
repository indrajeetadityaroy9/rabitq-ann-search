#pragma once

#include "../core/memory.hpp"
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <algorithm>

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
    
    // Support for learning/updating the matrix (Future ITQ hook)
    void set_matrix(const std::vector<float>& new_mat) {
        if (new_mat.size() != dim_ * dim_) {
            throw std::invalid_argument("Matrix size mismatch");
        }
        matrix_ = new_mat;
    }

private:
    size_t dim_;
    size_t padded_dim_;
    std::vector<float> matrix_; // D x D flattened

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