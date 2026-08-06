// bindings.mm
// PyTorch extension bindings for Luxar's 3D FP32 Metal Gaussian splatting backend.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <simd/simd.h>
#import <torch/extension.h>
#import <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <dlfcn.h>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <vector>

// ============================================================================
// Helper Types
// ============================================================================

// Matches Metal's uint3 ABI for setBytes constants.
struct uint3 {
    uint32_t x, y, z;
};

constexpr uint32_t kThreadgroupSize = 64;

// ============================================================================
// Metal Context Management
// ============================================================================

static std::string g_library_path;

void set_library_path(const std::string& path);

static NSArray<NSString*>* metalLibraryCandidatePaths() {
    NSMutableArray<NSString*>* paths = [NSMutableArray array];

    if (!g_library_path.empty()) {
        [paths addObject:[NSString stringWithUTF8String:g_library_path.c_str()]];
    }

    Dl_info info;
    if (dladdr((const void*)&metalLibraryCandidatePaths, &info) && info.dli_fname) {
        NSString* extensionPath = [NSString stringWithUTF8String:info.dli_fname];
        NSString* extensionDir = [extensionPath stringByDeletingLastPathComponent];
        [paths addObject:[extensionDir stringByAppendingPathComponent:@"src/default.metallib"]];
        [paths addObject:[extensionDir stringByAppendingPathComponent:@"default.metallib"]];
    }

    NSString* sourcePath = [NSString stringWithUTF8String:__FILE__];
    NSString* sourceDir = [sourcePath stringByDeletingLastPathComponent];
    [paths addObject:[sourceDir stringByAppendingPathComponent:@"default.metallib"]];
    [paths addObject:[sourceDir stringByAppendingPathComponent:@"src/default.metallib"]];

    return paths;
}

struct MetalContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    std::map<std::string, id<MTLComputePipelineState>> pipelines;

    MetalContext() : device(nil), queue(nil), library(nil) {
        // MET-4: exception-safe construction. The C++ destructor only runs
        // after a constructor returns normally, so any throw mid-init must
        // release whatever has already been retained — otherwise device /
        // queue accumulate +1 retain counts forever.
        @autoreleasepool {
            device = MTLCreateSystemDefaultDevice();
            if (!device) {
                throw std::runtime_error(
                    "Failed to create Metal device. Is this running on macOS with Metal support?");
            }

            queue = [device newCommandQueue];
            if (!queue) {
                [device release]; device = nil;
                throw std::runtime_error("Failed to create Metal command queue");
            }

            NSError* error = nil;
            NSMutableArray<NSString*>* attempted = [NSMutableArray array];

            for (NSString* libPath in metalLibraryCandidatePaths()) {
                [attempted addObject:libPath];
                if ([[NSFileManager defaultManager] fileExistsAtPath:libPath]) {
                    library = [device newLibraryWithFile:libPath error:&error];
                    if (library) {
                        break;
                    }
                }
            }

            if (!library) {
                NSString* msg = error ? [error localizedDescription] : @"Library not found";
                NSString* paths = [attempted componentsJoinedByString:@"\n  - "];
                std::string err = [[NSString stringWithFormat:
                    @"Failed to load Metal library: %@\nSearched:\n  - %@", msg, paths] UTF8String];
                [queue release];  queue = nil;
                [device release]; device = nil;
                throw std::runtime_error(err);
            }
        }
    }

    ~MetalContext() {
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

static MetalContext* g_ctx = nullptr;

// set_library_path is meant to be called once at extension import (see
// __init__.py), before any kernel dispatch. If a context already exists, the
// path may have been baked into pipelines that command buffers are using;
// silently destroying it could crash an in-flight dispatch. We therefore only
// honor a no-context call and treat re-set as a programming error.
void set_library_path(const std::string& path) {
    TORCH_CHECK(
        g_ctx == nullptr,
        "set_library_path() must be called before any Metal dispatch; "
        "a MetalContext has already been initialized."
    );
    g_library_path = path;
}

static MetalContext* metalContext() {
    if (!g_ctx) {
        g_ctx = new MetalContext();
    }
    return g_ctx;
}

// ============================================================================
// Tensor / Metal Buffer Helpers
// ============================================================================

id<MTLBuffer> tensorToMTLBuffer(const torch::Tensor& t) {
    TORCH_CHECK(t.device().is_mps(), "Tensor must be on MPS device");
    TORCH_CHECK(t.is_contiguous(), "Tensor must be contiguous");

    void* storage_ptr = t.storage().data_ptr().get();
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)storage_ptr;
    return buffer;
}

void setBufferWithOffset(
    id<MTLComputeCommandEncoder> enc,
    const torch::Tensor& t,
    int index
) {
    if (t.numel() == 0) {
        [enc setBuffer:nil offset:0 atIndex:index];
        return;
    }

    TORCH_CHECK(t.device().is_mps(),
        "Tensor at index ", index, " must be on MPS device, got ", t.device());
    TORCH_CHECK(t.is_contiguous(),
        "Tensor at index ", index, " must be contiguous. Use .contiguous() first.");

    id<MTLBuffer> buf = tensorToMTLBuffer(t);
    TORCH_CHECK(buf != nil, "Failed to get MTLBuffer for tensor at index ", index);

    NSUInteger offset = t.storage_offset() * t.element_size();
    NSUInteger buffer_size = [buf length];
    NSUInteger required_size = offset + (t.numel() * t.element_size());
    TORCH_CHECK(required_size <= buffer_size,
        "Buffer overflow: tensor at index ", index,
        " requires ", required_size, " bytes but buffer is ", buffer_size, " bytes");

    [enc setBuffer:buf offset:offset atIndex:index];
}

void checkCommandBuffer(id<MTLCommandBuffer> cmd, NSString* label) {
    if ([cmd status] == MTLCommandBufferStatusError) {
        NSString* errMsg = [[cmd error] localizedDescription];
        throw std::runtime_error([[NSString stringWithFormat:@"%@: %@", label, errMsg] UTF8String]);
    }
}

void validate_shape_3d(const std::vector<int64_t>& shape) {
    TORCH_CHECK(shape.size() == 3, "3D Metal kernels require shape (D, H, W)");
    TORCH_CHECK(shape[0] > 0 && shape[1] > 0 && shape[2] > 0,
        "shape dimensions must be positive");

    int64_t numel = shape[0] * shape[1] * shape[2];
    TORCH_CHECK(numel <= std::numeric_limits<uint32_t>::max(),
        "Metal 3D kernels currently support at most uint32_t output elements, got ", numel);

    // MET-3: per-splat AABB voxel counts inside the kernels are tracked in
    // 32-bit unsigned ints. A volume large enough to let a single wide splat
    // overflow that counter would silently drop the splat. Cap conservatively
    // at 2e9 voxels (well below UINT_MAX) and fail fast with a clear error.
    constexpr uint64_t kMaxVoxels = 2'000'000'000ULL;
    uint64_t total_voxels = static_cast<uint64_t>(shape[0])
                          * static_cast<uint64_t>(shape[1])
                          * static_cast<uint64_t>(shape[2]);
    TORCH_CHECK(total_voxels <= kMaxVoxels,
        "Volume D*H*W=", total_voxels,
        " exceeds Metal backend's per-splat AABB capacity (", kMaxVoxels, " voxels). "
        "This is a hard cap to prevent silent splat drops; reduce volume size or "
        "tile the workload.");
}

int64_t shape_numel(const std::vector<int64_t>& shape) {
    return shape[0] * shape[1] * shape[2];
}

void validate_splat_tensors_3d(
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const std::vector<int64_t>& shape
) {
    validate_shape_3d(shape);

    TORCH_CHECK(centers.device().is_mps() && conic.device().is_mps() && amps.device().is_mps(),
        "3D Metal tensors must be on MPS device");
    TORCH_CHECK(centers.scalar_type() == torch::kFloat32
            && conic.scalar_type() == torch::kFloat32
            && amps.scalar_type() == torch::kFloat32,
        "3D Metal tensors must be float32");
    TORCH_CHECK(centers.is_contiguous() && conic.is_contiguous() && amps.is_contiguous(),
        "3D Metal tensors must be contiguous");

    TORCH_CHECK(centers.dim() == 2 && centers.size(1) == 3,
        "centers must have shape (N, 3)");
    TORCH_CHECK(conic.dim() == 2 && conic.size(1) == 6,
        "conic must have shape (N, 6)");
    TORCH_CHECK(amps.dim() == 1, "amps must have shape (N,)");
    TORCH_CHECK(conic.size(0) == centers.size(0) && amps.size(0) == centers.size(0),
        "centers, conic, and amps batch dimensions must match");
    TORCH_CHECK(centers.size(0) <= std::numeric_limits<uint32_t>::max(),
        "Metal 3D kernels currently support at most uint32_t splats");
}

// ============================================================================
// Optional: Compute Conic from L in Metal
// ============================================================================

torch::Tensor compute_conic_metal(torch::Tensor Ls) {
    TORCH_CHECK(Ls.device().is_mps(), "Ls must be on MPS device");
    TORCH_CHECK(Ls.scalar_type() == torch::kFloat32, "Ls must be float32");
    TORCH_CHECK(Ls.is_contiguous(), "Ls must be contiguous");
    TORCH_CHECK(Ls.dim() == 3 && Ls.size(1) == 3 && Ls.size(2) == 3,
        "Ls must have shape (N, 3, 3)");
    TORCH_CHECK(Ls.size(0) <= std::numeric_limits<uint32_t>::max(),
        "compute_conic_metal supports at most uint32_t splats");

    auto conic = torch::empty({Ls.size(0), 6}, Ls.options().dtype(torch::kFloat32));
    if (Ls.size(0) == 0) {
        return conic;
    }

    // MET-1: wrap the dispatch body in an autorelease pool. Compiled with
    // -fno-objc-arc (manual reference counting), so MTLCommandBuffer /
    // MTLComputeCommandEncoder / NSString returned by Cocoa convenience
    // initializers would otherwise accumulate on the thread's outer pool —
    // a slow but real leak over thousands of training iterations.
    @autoreleasepool {
        MetalContext* ctx = metalContext();
        torch::mps::synchronize();

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ctx->getPipeline("compute_conic_from_L_3d")];

        setBufferWithOffset(enc, Ls, 0);
        setBufferWithOffset(enc, conic, 1);

        uint32_t n_splats = static_cast<uint32_t>(Ls.size(0));
        [enc setBytes:&n_splats length:sizeof(uint32_t) atIndex:2];

        MTLSize threads = MTLSizeMake(n_splats, 1, 1);
        MTLSize group = MTLSizeMake(std::min<uint32_t>(kThreadgroupSize, n_splats), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:group];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        checkCommandBuffer(cmd, @"compute_conic_from_L_3d failed");
    }

    return conic;
}

// ============================================================================
// Splat-Centric Forward Pass
// ============================================================================

torch::Tensor dispatch_forward_splat_3d(
    torch::Tensor centers,
    torch::Tensor conic,
    torch::Tensor amps,
    std::vector<int64_t> shape,
    float truncate,
    float intensity_floor
) {
    validate_splat_tensors_3d(centers, conic, amps, shape);

    // MET-10 (review-flagged minor): handle N=0 BEFORE allocating the output
    // tensor, avoiding a wasted allocation on the empty path.
    if (centers.size(0) == 0) {
        return torch::zeros(shape, centers.options().dtype(torch::kFloat32));
    }

    auto output = torch::empty(shape, centers.options().dtype(torch::kFloat32));
    int64_t total_pixels_i64 = shape_numel(shape);
    uint32_t total_pixels = static_cast<uint32_t>(total_pixels_i64);

    // MET-1: see compute_conic_metal — same MRC autorelease-pool rationale.
    @autoreleasepool {
        MetalContext* ctx = metalContext();
        torch::mps::synchronize();

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];

        // Zero the output in the same command buffer as the splat scatter pass.
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:ctx->getPipeline("zero_float_buffer")];
            setBufferWithOffset(enc, output, 0);
            [enc setBytes:&total_pixels length:sizeof(uint32_t) atIndex:1];
            MTLSize threads = MTLSizeMake(total_pixels, 1, 1);
            MTLSize group = MTLSizeMake(std::min<uint32_t>(kThreadgroupSize, total_pixels), 1, 1);
            [enc dispatchThreads:threads threadsPerThreadgroup:group];
            [enc endEncoding];
        }

        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:ctx->getPipeline("rasterize_forward_splat_centric_3d")];

            setBufferWithOffset(enc, centers, 0);
            setBufferWithOffset(enc, conic, 1);
            setBufferWithOffset(enc, amps, 2);
            setBufferWithOffset(enc, output, 3);

            uint3 shape_dhw = {
                static_cast<uint32_t>(shape[0]),
                static_cast<uint32_t>(shape[1]),
                static_cast<uint32_t>(shape[2]),
            };
            [enc setBytes:&shape_dhw length:sizeof(uint3) atIndex:4];

            uint32_t n_splats = static_cast<uint32_t>(centers.size(0));
            [enc setBytes:&n_splats length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:&truncate length:sizeof(float) atIndex:6];
            [enc setBytes:&intensity_floor length:sizeof(float) atIndex:7];

            MTLSize groups = MTLSizeMake(n_splats, 1, 1);
            MTLSize threadsPerGroup = MTLSizeMake(kThreadgroupSize, 1, 1);
            [enc dispatchThreadgroups:groups threadsPerThreadgroup:threadsPerGroup];
            [enc endEncoding];
        }

        [cmd commit];
        [cmd waitUntilCompleted];
        checkCommandBuffer(cmd, @"rasterize_forward_splat_centric_3d failed");
    }

    return output;
}

// ============================================================================
// Splat-Centric Backward Pass
// ============================================================================

std::vector<torch::Tensor> dispatch_backward_splat_3d(
    torch::Tensor grad_output,
    torch::Tensor centers,
    torch::Tensor conic,
    torch::Tensor amps,
    std::vector<int64_t> shape,
    float truncate,
    float intensity_floor
) {
    validate_splat_tensors_3d(centers, conic, amps, shape);
    TORCH_CHECK(grad_output.device().is_mps(), "grad_output must be on MPS device");
    TORCH_CHECK(grad_output.scalar_type() == torch::kFloat32, "grad_output must be float32");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(grad_output.dim() == 3
            && grad_output.size(0) == shape[0]
            && grad_output.size(1) == shape[1]
            && grad_output.size(2) == shape[2],
        "grad_output shape must match the provided 3D shape");

    int64_t N = centers.size(0);
    auto opts = centers.options().dtype(torch::kFloat32);
    if (N == 0) {
        return {
            torch::zeros({N, 3}, opts),
            torch::zeros({N, 6}, opts),
            torch::zeros({N}, opts),
        };
    }
    auto d_centers = torch::empty({N, 3}, opts);
    auto d_conic = torch::empty({N, 6}, opts);
    auto d_amps = torch::empty({N}, opts);

    // MET-1: see compute_conic_metal — same MRC autorelease-pool rationale.
    @autoreleasepool {
        MetalContext* ctx = metalContext();
        torch::mps::synchronize();

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ctx->getPipeline("rasterize_backward_splat_centric_3d")];

        setBufferWithOffset(enc, grad_output, 0);
        setBufferWithOffset(enc, centers, 1);
        setBufferWithOffset(enc, conic, 2);
        setBufferWithOffset(enc, amps, 3);
        setBufferWithOffset(enc, d_centers, 4);
        setBufferWithOffset(enc, d_conic, 5);
        setBufferWithOffset(enc, d_amps, 6);

        uint3 shape_dhw = {
            static_cast<uint32_t>(shape[0]),
            static_cast<uint32_t>(shape[1]),
            static_cast<uint32_t>(shape[2]),
        };
        [enc setBytes:&shape_dhw length:sizeof(uint3) atIndex:7];

        uint32_t n_splats = static_cast<uint32_t>(N);
        [enc setBytes:&n_splats length:sizeof(uint32_t) atIndex:8];
        [enc setBytes:&truncate length:sizeof(float) atIndex:9];
        [enc setBytes:&intensity_floor length:sizeof(float) atIndex:10];

        MTLSize groups = MTLSizeMake(n_splats, 1, 1);
        MTLSize threadsPerGroup = MTLSizeMake(kThreadgroupSize, 1, 1);
        [enc dispatchThreadgroups:groups threadsPerThreadgroup:threadsPerGroup];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        checkCommandBuffer(cmd, @"rasterize_backward_splat_centric_3d failed");
    }

    return {d_centers, d_conic, d_amps};
}

// ============================================================================
// PyBind11 Module
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Splat-centric Metal Gaussian splatting backend for Luxar";

    m.def("set_library_path", &set_library_path,
          "Set the path to default.metallib used by the Metal kernels");

    m.def("compute_conic_metal", &compute_conic_metal,
          "Compute 3D packed conic (Σ⁻¹) from Cholesky factors in [Z,Y,X] order");

    m.def("forward_splat_3d", &dispatch_forward_splat_3d,
          "Splat-centric 3D Metal forward pass");

    m.def("backward_splat_3d", &dispatch_backward_splat_3d,
          "Splat-centric 3D Metal backward pass");
}
