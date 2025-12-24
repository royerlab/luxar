// bindings.mm
// C++ dispatcher for Metal-accelerated Gaussian splatting
//
// This file provides Python bindings for Metal compute kernels.
// It handles Metal device/queue management, buffer allocation, and kernel dispatch.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <simd/simd.h>
#import <torch/extension.h>
#import <torch/torch.h>

#include <map>
#include <string>
#include <vector>

// ============================================================================
// Helper Types
// ============================================================================

// uint3 struct for passing to Metal kernels (matches Metal's uint3 layout)
struct uint3 {
    uint32_t x, y, z;
};

// ============================================================================
// Metal Context Management
// ============================================================================

struct MetalContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    std::map<std::string, id<MTLComputePipelineState>> pipelines;

    MetalContext() {
        // Get default Metal device
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            throw std::runtime_error("Failed to create Metal device. Is this running on macOS with Metal support?");
        }

        // Create command queue
        queue = [device newCommandQueue];
        if (!queue) {
            throw std::runtime_error("Failed to create Metal command queue");
        }

        // Load precompiled Metal library
        // Try multiple locations to find the library
        NSArray* searchPaths = @[
            // Relative to source directory (build time and runtime)
            [[NSString stringWithUTF8String:__FILE__] stringByDeletingLastPathComponent],
            // Current directory (fallback)
            [[NSFileManager defaultManager] currentDirectoryPath]
        ];

        NSError* error = nil;
        NSString* foundPath = nil;

        for (NSString* searchDir in searchPaths) {
            NSString* libPath = [searchDir stringByAppendingPathComponent:@"default.metallib"];
            if ([[NSFileManager defaultManager] fileExistsAtPath:libPath]) {
                library = [device newLibraryWithFile:libPath error:&error];
                if (library) {
                    foundPath = libPath;
                    break;
                }
            }
        }

        if (!library) {
            NSString* msg = error ? [error localizedDescription] : @"Library not found in search paths";
            throw std::runtime_error([[NSString stringWithFormat:@"Failed to load Metal library: %@", msg] UTF8String]);
        }
    }

    ~MetalContext() {
        // Release Metal objects (ARC disabled)
        for (auto& pair : pipelines) {
            [pair.second release];
        }
        [library release];
        [queue release];
        [device release];
    }

    id<MTLComputePipelineState> getPipeline(const char* name) {
        auto it = pipelines.find(name);
        if (it != pipelines.end()) return it->second;

        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
        if (!func) {
            throw std::runtime_error(std::string("Missing kernel function: ") + name);
        }

        NSError* error = nil;
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:func error:&error];
        [func release];

        if (!pso) {
            NSString* msg = error ? [error localizedDescription] : @"Unknown error";
            throw std::runtime_error([[NSString stringWithFormat:@"Failed to create pipeline for %s: %@", name, msg] UTF8String]);
        }

        pipelines[name] = pso;
        return pso;
    }
};

// Global context (lazy initialization)
static MetalContext* g_ctx = nullptr;

// ============================================================================
// Helper Functions
// ============================================================================

// Get MTLBuffer from PyTorch MPS tensor
// The STORAGE's data_ptr() is the id<MTLBuffer> on MPS
// (not the tensor's data_ptr(), which includes offset)
id<MTLBuffer> tensorToMTLBuffer(const torch::Tensor& t) {
    TORCH_CHECK(t.device().is_mps(), "Tensor must be on MPS device");
    TORCH_CHECK(t.is_contiguous(), "Tensor must be contiguous");

    // Get the storage's data pointer, which is the MTLBuffer
    void* storage_ptr = t.storage().data_ptr().get();

    // Cast to MTLBuffer (MPS tensors are backed by MTLBuffers)
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)storage_ptr;

    return buffer;
}

// Set buffer with correct offset handling
void setBufferWithOffset(
    id<MTLComputeCommandEncoder> enc,
    const torch::Tensor& t,
    int index
) {
    // Handle empty tensors
    if (t.numel() == 0) {
        [enc setBuffer:nil offset:0 atIndex:index];
        return;
    }

    // Validate tensor
    TORCH_CHECK(t.device().is_mps(),
        "Tensor at index ", index, " must be on MPS device, got ", t.device());
    TORCH_CHECK(t.is_contiguous(),
        "Tensor at index ", index, " must be contiguous. Use .contiguous() first.");

    // Get MTLBuffer from storage
    id<MTLBuffer> buf = tensorToMTLBuffer(t);
    TORCH_CHECK(buf != nil, "Failed to get MTLBuffer for tensor at index ", index);

    // Compute byte offset
    NSUInteger offset = t.storage_offset() * t.element_size();

    // Validate offset doesn't exceed buffer bounds
    NSUInteger buffer_size = [buf length];
    NSUInteger required_size = offset + (t.numel() * t.element_size());
    TORCH_CHECK(required_size <= buffer_size,
        "Buffer overflow: tensor at index ", index,
        " requires ", required_size, " bytes but buffer is ", buffer_size, " bytes");

    [enc setBuffer:buf offset:offset atIndex:index];
}

// ============================================================================
// Optional: Compute Conic from L in Metal (faster alternative)
// ============================================================================

torch::Tensor compute_conic_metal(torch::Tensor Ls) {
    if (!g_ctx) g_ctx = new MetalContext();

    int N = Ls.size(0);
    auto conic = torch::zeros({N, 6}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMPS));

    torch::mps::synchronize();

    id<MTLCommandBuffer> cmd = [g_ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ctx->getPipeline("compute_conic_from_L_3d")];

    setBufferWithOffset(enc, Ls, 0);
    setBufferWithOffset(enc, conic, 1);

    uint32_t n_splats = (uint32_t)N;
    [enc setBytes:&n_splats length:sizeof(uint32_t) atIndex:2];

    MTLSize gridSize = MTLSizeMake(N, 1, 1);
    MTLSize groupSize = MTLSizeMake(std::min(256, std::max(1, N)), 1, 1);
    [enc dispatchThreadgroups:MTLSizeMake((N + 255) / 256, 1, 1)
          threadsPerThreadgroup:groupSize];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    if ([cmd status] == MTLCommandBufferStatusError) {
        NSString* errMsg = [[cmd error] localizedDescription];
        throw std::runtime_error([[NSString stringWithFormat:@"compute_conic_from_L_3d failed: %@", errMsg] UTF8String]);
    }

    return conic;
}

// ============================================================================
// Forward Pass (3D with tiling)
// ============================================================================

std::vector<torch::Tensor> dispatch_forward_3d(
    torch::Tensor centers,   // (N, 3)
    torch::Tensor conic,     // (N, 6) - precomputed in PyTorch
    torch::Tensor amps,      // (N,)
    torch::Tensor sharpness, // (N,)
    torch::Tensor Ls,        // (N, 3, 3) - needed for sigma_diag in binning
    std::vector<int64_t> shape,  // [D, H, W]
    float truncate,
    float intensity_floor,
    int tile_size            // Configurable tile size (e.g., 4 or 8)
) {
    if (!g_ctx) g_ctx = new MetalContext();

    int N = centers.size(0);
    int D = shape[0], H = shape[1], W = shape[2];

    // Grid dimensions based on configurable tile size
    // NOTE: Metal uint3 uses (x,y,z) = (W,H,D) order
    uint32_t grid_x = (W + tile_size - 1) / tile_size;
    uint32_t grid_y = (H + tile_size - 1) / tile_size;
    uint32_t grid_z = (D + tile_size - 1) / tile_size;
    int num_tiles = grid_x * grid_y * grid_z;

    // Allocate buffers
    auto tile_counts = torch::zeros({num_tiles}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kMPS));
    auto output = torch::zeros(shape, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMPS));

    // CRITICAL: Synchronize MPS before Metal kernel dispatch
    // This ensures MPS tensor allocation is complete before Metal accesses them
    torch::mps::synchronize();

    // === Pass 1: Preprocess (count splats per tile) ===
    {
        id<MTLCommandBuffer> cmd = [g_ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_ctx->getPipeline("preprocess_3d")];

        setBufferWithOffset(enc, centers, 0);
        setBufferWithOffset(enc, Ls, 1);
        setBufferWithOffset(enc, sharpness, 2);
        setBufferWithOffset(enc, tile_counts, 3);

        [enc setBytes:&truncate length:sizeof(float) atIndex:4];

        uint3 grid_dims_metal = {grid_x, grid_y, grid_z};
        [enc setBytes:&grid_dims_metal length:sizeof(uint3) atIndex:5];

        uint32_t n_splats = (uint32_t)N;
        [enc setBytes:&n_splats length:sizeof(uint32_t) atIndex:6];

        int32_t tile_size_param = (int32_t)tile_size;
        [enc setBytes:&tile_size_param length:sizeof(int32_t) atIndex:7];

        // CRITICAL FIX: Use proper threadgroup sizing
        // With gridSize=(1,1,1) and groupSize=(1,1,1), some Metal GPUs may not dispatch
        // Use at least 32 threads per group (one SIMD group) or dispatch as threadgroups
        MTLSize gridSize = MTLSizeMake(N, 1, 1);
        MTLSize groupSize = MTLSizeMake(std::min(64, std::max(1, N)), 1, 1);
        [enc dispatchThreadgroups:MTLSizeMake((N + 63) / 64, 1, 1)
              threadsPerThreadgroup:groupSize];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        // Check for Metal errors
        if ([cmd status] == MTLCommandBufferStatusError) {
            NSString* errMsg = [[cmd error] localizedDescription];
            throw std::runtime_error([[NSString stringWithFormat:@"preprocess_3d failed: %@", errMsg] UTF8String]);
        }
    }

    // === Pass 2: Prefix sum (GPU via PyTorch cumsum) ===
    auto zeros = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kMPS));
    auto counts_padded = torch::cat({zeros, tile_counts.slice(0, 0, num_tiles - 1)});
    auto tile_offsets = torch::cumsum(counts_padded, 0, torch::kInt32);

    // Get total (this forces a CPU sync since we call .item())
    int total = (tile_offsets[-1] + tile_counts[-1]).item<int>();

    auto tile_content = torch::zeros({std::max(total, 1)}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kMPS));
    auto tile_write_heads = torch::zeros({num_tiles}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kMPS));

    // Sync before Metal kernels (ensures PyTorch MPS ops complete)
    // NOTE: The .item() call above already forced a sync, but explicit is safer
    torch::mps::synchronize();

    // === Pass 3: Binning ===
    {
        id<MTLCommandBuffer> cmd = [g_ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_ctx->getPipeline("bin_3d")];

        setBufferWithOffset(enc, centers, 0);
        setBufferWithOffset(enc, Ls, 1);
        setBufferWithOffset(enc, sharpness, 2);
        setBufferWithOffset(enc, tile_offsets, 3);
        setBufferWithOffset(enc, tile_write_heads, 4);
        setBufferWithOffset(enc, tile_content, 5);
        [enc setBytes:&truncate length:sizeof(float) atIndex:6];

        uint3 grid_dims_metal = {grid_x, grid_y, grid_z};
        [enc setBytes:&grid_dims_metal length:sizeof(uint3) atIndex:7];

        uint32_t n_splats = (uint32_t)N;
        [enc setBytes:&n_splats length:sizeof(uint32_t) atIndex:8];

        int32_t tile_size_param = (int32_t)tile_size;
        [enc setBytes:&tile_size_param length:sizeof(int32_t) atIndex:9];

        MTLSize gridSize = MTLSizeMake(N, 1, 1);
        MTLSize groupSize = MTLSizeMake(std::min(256, N), 1, 1);
        [enc dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    // === Pass 4: Rasterization ===
    {
        id<MTLCommandBuffer> cmd = [g_ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_ctx->getPipeline("rasterize_fwd_3d")];

        setBufferWithOffset(enc, centers, 0);
        setBufferWithOffset(enc, conic, 1);
        setBufferWithOffset(enc, amps, 2);
        setBufferWithOffset(enc, sharpness, 3);
        setBufferWithOffset(enc, tile_offsets, 4);
        setBufferWithOffset(enc, tile_counts, 5);
        setBufferWithOffset(enc, tile_content, 6);
        setBufferWithOffset(enc, output, 7);

        uint3 img_size = {(uint32_t)W, (uint32_t)H, (uint32_t)D};
        [enc setBytes:&img_size length:sizeof(uint3) atIndex:8];

        uint3 grid_dims_metal = {grid_x, grid_y, grid_z};
        [enc setBytes:&grid_dims_metal length:sizeof(uint3) atIndex:9];

        [enc setBytes:&truncate length:sizeof(float) atIndex:10];
        [enc setBytes:&intensity_floor length:sizeof(float) atIndex:11];

        // CRITICAL: Pad grid to multiple of tile size for full threadgroups
        int W_padded = ((W + tile_size - 1) / tile_size) * tile_size;
        int H_padded = ((H + tile_size - 1) / tile_size) * tile_size;
        int D_padded = ((D + tile_size - 1) / tile_size) * tile_size;

        MTLSize gridSize = MTLSizeMake(W_padded, H_padded, D_padded);
        MTLSize groupSize = MTLSizeMake(tile_size, tile_size, tile_size);
        [enc dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    // Return output AND intermediate buffers for backward pass
    return {output, tile_counts, tile_offsets, tile_content};
}

// ============================================================================
// Backward Pass (3D with tiling and SIMD reduction)
// ============================================================================

std::vector<torch::Tensor> dispatch_backward_3d(
    torch::Tensor grad_output,    // (D, H, W)
    torch::Tensor centers,        // (N, 3)
    torch::Tensor conic,          // (N, 6)
    torch::Tensor amps,           // (N,)
    torch::Tensor sharpness,      // (N,)
    torch::Tensor tile_offsets,   // From forward pass
    torch::Tensor tile_counts,    // From forward pass
    torch::Tensor tile_content,   // From forward pass
    std::vector<int64_t> shape,   // [D, H, W]
    float truncate,
    float intensity_floor,
    int tile_size                 // Must match forward pass!
) {
    if (!g_ctx) g_ctx = new MetalContext();

    int N = centers.size(0);
    int D = shape[0], H = shape[1], W = shape[2];

    // Grid dimensions (must match forward!)
    uint32_t grid_x = (W + tile_size - 1) / tile_size;
    uint32_t grid_y = (H + tile_size - 1) / tile_size;
    uint32_t grid_z = (D + tile_size - 1) / tile_size;

    // CRITICAL: Zero-initialize gradient buffers
    auto d_centers = torch::zeros({N, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMPS));
    auto d_conic = torch::zeros({N, 6}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMPS));
    auto d_amps = torch::zeros({N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMPS));
    auto d_sharpness = torch::zeros({N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMPS));

    // Ensure MPS operations complete
    torch::mps::synchronize();

    // === Backward Rasterization Kernel ===
    {
        id<MTLCommandBuffer> cmd = [g_ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_ctx->getPipeline("rasterize_bwd_3d")];

        // Input buffers
        setBufferWithOffset(enc, grad_output, 0);
        setBufferWithOffset(enc, centers, 1);
        setBufferWithOffset(enc, conic, 2);
        setBufferWithOffset(enc, amps, 3);
        setBufferWithOffset(enc, sharpness, 4);
        setBufferWithOffset(enc, tile_offsets, 5);
        setBufferWithOffset(enc, tile_counts, 6);
        setBufferWithOffset(enc, tile_content, 7);

        // Output gradient buffers
        setBufferWithOffset(enc, d_centers, 8);
        setBufferWithOffset(enc, d_conic, 9);
        setBufferWithOffset(enc, d_amps, 10);
        setBufferWithOffset(enc, d_sharpness, 11);

        // Constants
        uint3 img_size = {(uint32_t)W, (uint32_t)H, (uint32_t)D};
        [enc setBytes:&img_size length:sizeof(uint3) atIndex:12];

        uint3 grid_dims_metal = {grid_x, grid_y, grid_z};
        [enc setBytes:&grid_dims_metal length:sizeof(uint3) atIndex:13];

        [enc setBytes:&truncate length:sizeof(float) atIndex:14];
        [enc setBytes:&intensity_floor length:sizeof(float) atIndex:15];

        // CRITICAL: Pad grid to multiple of tile size for full threadgroups
        int W_padded = ((W + tile_size - 1) / tile_size) * tile_size;
        int H_padded = ((H + tile_size - 1) / tile_size) * tile_size;
        int D_padded = ((D + tile_size - 1) / tile_size) * tile_size;

        MTLSize gridSize = MTLSizeMake(W_padded, H_padded, D_padded);
        MTLSize groupSize = MTLSizeMake(tile_size, tile_size, tile_size);
        [enc dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    return {d_centers, d_conic, d_amps, d_sharpness};
}

// ============================================================================
// Forward Pass (nD generic)
// ============================================================================

std::vector<torch::Tensor> dispatch_forward_nd(
    torch::Tensor centers,        // (N, dim)
    torch::Tensor Ls,             // (N, dim, dim)
    torch::Tensor amps,           // (N,)
    torch::Tensor sharpness,      // (N,)
    std::vector<int64_t> shape,   // [s_0, s_1, ..., s_{dim-1}]
    float truncate,
    float intensity_floor
) {
    if (!g_ctx) g_ctx = new MetalContext();

    int N = centers.size(0);
    int dim = centers.size(1);

    TORCH_CHECK(dim <= 8, "Metal nD kernel supports up to 8 dimensions");
    TORCH_CHECK(shape.size() == dim, "Shape must match dimensionality");

    // Compute total size
    int64_t total_size = 1;
    for (int i = 0; i < dim; i++) total_size *= shape[i];

    auto output = torch::zeros(shape, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMPS));

    torch::mps::synchronize();

    // === Rasterization Kernel ===
    {
        id<MTLCommandBuffer> cmd = [g_ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_ctx->getPipeline("rasterize_fwd_nd")];

        setBufferWithOffset(enc, centers, 0);
        setBufferWithOffset(enc, Ls, 1);
        setBufferWithOffset(enc, amps, 2);
        setBufferWithOffset(enc, sharpness, 3);
        setBufferWithOffset(enc, output, 4);

        uint32_t n_splats = (uint32_t)N;
        [enc setBytes:&n_splats length:sizeof(uint32_t) atIndex:5];

        uint32_t dim_u = (uint32_t)dim;
        [enc setBytes:&dim_u length:sizeof(uint32_t) atIndex:6];

        // Pass shape as array
        uint32_t shape_arr[8] = {0};
        for (int i = 0; i < dim; i++) shape_arr[i] = (uint32_t)shape[i];
        [enc setBytes:shape_arr length:sizeof(shape_arr) atIndex:7];

        [enc setBytes:&truncate length:sizeof(float) atIndex:8];
        [enc setBytes:&intensity_floor length:sizeof(float) atIndex:9];

        MTLSize gridSize = MTLSizeMake(total_size, 1, 1);
        MTLSize groupSize = MTLSizeMake(std::min((int64_t)256, total_size), 1, 1);
        [enc dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    return {output};
}

// ============================================================================
// Backward Pass (nD generic with SIMD reduction)
// ============================================================================

std::vector<torch::Tensor> dispatch_backward_nd(
    torch::Tensor grad_output,    // Shape matches forward output
    torch::Tensor centers,        // (N, dim)
    torch::Tensor Ls,             // (N, dim, dim)
    torch::Tensor amps,           // (N,)
    torch::Tensor sharpness,      // (N,)
    std::vector<int64_t> shape,
    float truncate,
    float intensity_floor
) {
    if (!g_ctx) g_ctx = new MetalContext();

    int N = centers.size(0);
    int dim = centers.size(1);

    TORCH_CHECK(dim <= 8, "Metal nD kernel supports up to 8 dimensions");

    // Compute total size
    int64_t total_size = 1;
    for (int i = 0; i < dim; i++) total_size *= shape[i];

    // CRITICAL: Zero-initialize gradient buffers
    auto d_centers = torch::zeros({N, dim}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMPS));
    auto d_Ls = torch::zeros({N, dim, dim}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMPS));
    auto d_amps = torch::zeros({N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMPS));
    auto d_sharpness = torch::zeros({N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMPS));

    torch::mps::synchronize();

    // === Backward Kernel ===
    {
        id<MTLCommandBuffer> cmd = [g_ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_ctx->getPipeline("rasterize_bwd_nd")];

        setBufferWithOffset(enc, grad_output, 0);
        setBufferWithOffset(enc, centers, 1);
        setBufferWithOffset(enc, Ls, 2);
        setBufferWithOffset(enc, amps, 3);
        setBufferWithOffset(enc, sharpness, 4);

        // Output gradient buffers
        setBufferWithOffset(enc, d_centers, 5);
        setBufferWithOffset(enc, d_Ls, 6);
        setBufferWithOffset(enc, d_amps, 7);
        setBufferWithOffset(enc, d_sharpness, 8);

        uint32_t n_splats = (uint32_t)N;
        [enc setBytes:&n_splats length:sizeof(uint32_t) atIndex:9];

        uint32_t dim_u = (uint32_t)dim;
        [enc setBytes:&dim_u length:sizeof(uint32_t) atIndex:10];

        // Shape and constants
        uint32_t shape_arr[8] = {0};
        for (int i = 0; i < dim; i++) shape_arr[i] = (uint32_t)shape[i];
        [enc setBytes:shape_arr length:sizeof(shape_arr) atIndex:11];

        [enc setBytes:&truncate length:sizeof(float) atIndex:12];
        [enc setBytes:&intensity_floor length:sizeof(float) atIndex:13];

        MTLSize gridSize = MTLSizeMake(total_size, 1, 1);
        MTLSize groupSize = MTLSizeMake(std::min((int64_t)256, total_size), 1, 1);
        [enc dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    return {d_centers, d_Ls, d_amps, d_sharpness};
}

// ============================================================================
// PyBind11 Module
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Metal-accelerated Gaussian splatting backend for Luxar";

    m.def("compute_conic_metal", &compute_conic_metal,
          "Compute conic (Σ⁻¹) from L in Metal - faster alternative to PyTorch");

    m.def("forward_3d", &dispatch_forward_3d,
          "Metal forward pass (3D tiled) - returns [output, tile_counts, tile_offsets, tile_content]");

    m.def("backward_3d", &dispatch_backward_3d,
          "Metal backward pass (3D tiled with SIMD reduction) - returns [d_centers, d_conic, d_amps, d_sharpness]");

    m.def("forward_nd", &dispatch_forward_nd,
          "Metal forward pass (generic nD) - returns [output]");

    m.def("backward_nd", &dispatch_backward_nd,
          "Metal backward pass (generic nD with SIMD reduction) - returns [d_centers, d_Ls, d_amps, d_sharpness]");
}
