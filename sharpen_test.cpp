#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <iostream>

using namespace cv;


// ============================================================================
// OpenCL implementation of computeMaxDiffMatrix
//
// OpenCL version compatibility:
//   - Designed to compile with OpenCL 1.2 headers.
//   - Uses clCreateCommandQueueWithProperties dynamically when the runtime
//     provides it.
//   - Falls back to clCreateCommandQueue for older implementations.
//
// The GPU performs the expensive histogram accumulation.
// Histogram reduction and the final M/P construction remain on the CPU,
// preserving the semantics of the existing implementation.
//
// ============================================================================

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

//#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace MaxDiffOpenCL
{

    using namespace cv;

    // -----------------------------------------------------------------------------
    // OpenCL error helper
    // -----------------------------------------------------------------------------

    static const char* clErrorString(cl_int err)
    {
        switch (err)
        {
        case CL_SUCCESS:                                   return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:                          return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:                      return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:                    return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:             return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES:                           return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:                         return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:              return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP:                           return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH:                     return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:                return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE:                     return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE:                               return "CL_MAP_FAILURE";
        case CL_INVALID_VALUE:                             return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE:                       return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM:                          return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:                            return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:                           return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:                  return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE:                     return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR:                          return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT:                        return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:           return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE:                         return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER:                           return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY:                            return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS:                     return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM:                           return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:                return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:                       return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:                 return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL:                            return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:                          return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:                          return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:                           return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:                        return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:                     return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:                    return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:                     return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET:                      return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST:                    return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT:                              return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION:                          return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT:                          return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE:                        return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL:                          return "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE:                   return "CL_INVALID_GLOBAL_WORK_SIZE";
        default:                                            return "CL_UNKNOWN_ERROR";
        }
    }

    static void checkCL(cl_int err, const char* what)
    {
        if (err != CL_SUCCESS)
        {
            throw std::runtime_error(
                std::string(what) +
                " failed: " +
                clErrorString(err) +
                " (" +
                std::to_string(err) +
                ")");
        }
    }

    // -----------------------------------------------------------------------------
    // Build log
    //
    // IMPORTANT:
    // Do not use result.data() here. Older C++/OpenCL header combinations can
    // expose the argument as void* while std::string::data() is const char*.
    // &result[0] is writable for a non-empty std::string.
    // -----------------------------------------------------------------------------

    static std::string getBuildLog(cl_program program, cl_device_id device)
    {
        size_t size = 0;

        cl_int err = clGetProgramBuildInfo(
            program,
            device,
            CL_PROGRAM_BUILD_LOG,
            0,
            nullptr,
            &size);

        if (err != CL_SUCCESS || size == 0)
            return std::string();

        std::string result(size, '\0');

        err = clGetProgramBuildInfo(
            program,
            device,
            CL_PROGRAM_BUILD_LOG,
            size,
            &result[0],
            nullptr);

        if (err != CL_SUCCESS)
            return std::string();

        return result;
    }

    // -----------------------------------------------------------------------------
    // Dynamically obtain clCreateCommandQueueWithProperties.
    //
    // This is the important compatibility trick.
    //
    // The source can therefore be compiled against OpenCL 1.2 headers, where the
    // function declaration may not exist, while still using the newer API when the
    // runtime provides it.
    //
    // If the runtime does not expose it, the code falls back to the universally
    // available OpenCL 1.2 clCreateCommandQueue.
    // -----------------------------------------------------------------------------

    typedef cl_command_queue(CL_API_CALL* PFN_clCreateCommandQueueWithPropertiesCompat)(
        cl_context,
        cl_device_id,
        const cl_queue_properties*,
        cl_int*);

    static cl_command_queue createCommandQueueCompat(
        cl_context context,
        cl_device_id device,
        cl_platform_id platform)
    {
        cl_int err = CL_SUCCESS;

        PFN_clCreateCommandQueueWithPropertiesCompat createWithProperties =
            reinterpret_cast<PFN_clCreateCommandQueueWithPropertiesCompat>(
                clGetExtensionFunctionAddressForPlatform(
                    platform,
                    "clCreateCommandQueueWithProperties"));

        if (createWithProperties)
        {
            const cl_queue_properties properties[] = {
                0
            };

            cl_command_queue queue =
                createWithProperties(
                    context,
                    device,
                    properties,
                    &err);

            if (err == CL_SUCCESS && queue)
                return queue;
        }

        // OpenCL 1.2 fallback.
        //
        // This API is deprecated in newer headers but remains the compatibility
        // path for older OpenCL implementations.
        //
        // We intentionally do not pass profiling or other optional properties.
        return clCreateCommandQueue(
            context,
            device,
            0,
            &err);
    }

    // -----------------------------------------------------------------------------
    // Offset description
    // -----------------------------------------------------------------------------

    struct Offset
    {
        int dx;
        int dy;
    };

    // -----------------------------------------------------------------------------
    // Generate offsets in EXACTLY the same order as the CPU implementation.
    //
    // Existing CPU code:
    //
    //     int dx = 1;
    //
    //     for (int dy = 0; dy <= radius; ++dy)
    //     {
    //         ...
    //         for (; dx <= radius; ++dx)
    //         {
    //             ...
    //         }
    //
    //         dx = -radius;
    //     }
    //
    // Therefore:
    //
    //   dy = 0 : dx = 1 ... radius
    //   dy = 1 : dx = -radius ... radius
    //   dy = 2 : dx = -radius ... radius
    //   ...
    //   dy = radius : dx = -radius ... radius
    //
    // This gives:
    //     radius + radius * 2 + radius
    //   = (radius + 1) * (2 * radius)
    //   = (radius + 1) * (K - 1)
    // -----------------------------------------------------------------------------

    static std::vector<Offset> makeOffsets(int radius)
    {
        std::vector<Offset> offsets;

        offsets.reserve(
            static_cast<size_t>(2 * radius * (radius + 1)));

        for (int dy = 0; dy <= radius; ++dy)
        {
            if (dy == 0)
            {
                for (int dx = 1; dx <= radius; ++dx)
                    offsets.push_back({ dx, dy });
            }
            else
            {
                for (int dx = -radius; dx <= radius; ++dx)
                    offsets.push_back({ dx, dy });
            }
        }

        return offsets;
    }

    // -----------------------------------------------------------------------------
    // OpenCL context
    // -----------------------------------------------------------------------------

    struct Context
    {
        cl_platform_id platform = nullptr;
        cl_device_id device = nullptr;
        cl_context context = nullptr;
        cl_command_queue queue = nullptr;
        cl_program program = nullptr;
        cl_kernel kernel = nullptr;

        ~Context()
        {
            if (kernel)
                clReleaseKernel(kernel);

            if (program)
                clReleaseProgram(program);

            if (queue)
                clReleaseCommandQueue(queue);

            if (context)
                clReleaseContext(context);
        }
    };

    // -----------------------------------------------------------------------------
    // Select a GPU device.
    //
    // If no GPU is available, fall back to any OpenCL device.
    //
    // This keeps the function usable on systems where the OpenCL implementation
    // does not expose a GPU device.
    // -----------------------------------------------------------------------------

    static bool chooseDevice(
        cl_platform_id& selectedPlatform,
        cl_device_id& selectedDevice)
    {
        selectedPlatform = nullptr;
        selectedDevice = nullptr;

        cl_uint platformCount = 0;

        cl_int err = clGetPlatformIDs(
            0,
            nullptr,
            &platformCount);

        if (err != CL_SUCCESS || platformCount == 0)
            return false;

        std::vector<cl_platform_id> platforms(platformCount);

        err = clGetPlatformIDs(
            platformCount,
            platforms.data(),
            nullptr);

        if (err != CL_SUCCESS)
            return false;

        // First pass: GPU.
        for (cl_platform_id platform : platforms)
        {
            cl_uint deviceCount = 0;

            err = clGetDeviceIDs(
                platform,
                CL_DEVICE_TYPE_GPU,
                0,
                nullptr,
                &deviceCount);

            if (err != CL_SUCCESS || deviceCount == 0)
                continue;

            std::vector<cl_device_id> devices(deviceCount);

            err = clGetDeviceIDs(
                platform,
                CL_DEVICE_TYPE_GPU,
                deviceCount,
                devices.data(),
                nullptr);

            if (err == CL_SUCCESS && !devices.empty())
            {
                selectedPlatform = platform;
                selectedDevice = devices[0];
                return true;
            }
        }

        // Second pass: any device.
        for (cl_platform_id platform : platforms)
        {
            cl_uint deviceCount = 0;

            err = clGetDeviceIDs(
                platform,
                CL_DEVICE_TYPE_ALL,
                0,
                nullptr,
                &deviceCount);

            if (err != CL_SUCCESS || deviceCount == 0)
                continue;

            std::vector<cl_device_id> devices(deviceCount);

            err = clGetDeviceIDs(
                platform,
                CL_DEVICE_TYPE_ALL,
                deviceCount,
                devices.data(),
                nullptr);

            if (err == CL_SUCCESS && !devices.empty())
            {
                selectedPlatform = platform;
                selectedDevice = devices[0];
                return true;
            }
        }

        return false;
    }

    // -----------------------------------------------------------------------------
    // Kernel
    //
    // One work-group computes one partial 256-bin histogram.
    //
    // Work-groups are distributed over:
    //
    //     offsetIndex * groupCount + groupIndex
    //
    // Each group processes a strided subset of valid pixels.
    //
    // A local histogram is used to avoid global atomic contention.
    //
    // Only OpenCL 1.2 functionality is used.
    // -----------------------------------------------------------------------------

    static const char* kernelSource = R"CLC(

__kernel void buildPartialHistograms(
    __global const uchar* image,
    const int width,
    const int height,

    const int xStart,
    const int yStart,
    const int validWidth,
    const int validHeight,

    const int radius,

    __global const int2* offsets,
    const int offsetCount,

    const int groupCount,

    __global uint* partialHistograms)
{
    __local uint histogram[256];

    const uint localId = get_local_id(0);
    const uint localSize = get_local_size(0);

    // Clear the local histogram.
    for (uint i = localId; i < 256; i += localSize)
        histogram[i] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint groupId = get_group_id(0);

    const uint offsetIndex = groupId / (uint)groupCount;
    const uint subgroupIndex = groupId % (uint)groupCount;

    if (offsetIndex >= (uint)offsetCount)
        return;

    const int2 offset = offsets[offsetIndex];

    const int totalPixels = validWidth * validHeight;

    // Work is distributed between work-groups belonging to the same offset.
    //
    // This avoids having one giant work-group for a large image and also
    // produces several independent histograms that can be reduced on CPU.
    for (
        int p = (int)subgroupIndex * (int)get_local_size(0)
              + (int)localId;
        p < totalPixels;
        p += groupCount * (int)get_local_size(0))
    {
        const int y = p / validWidth + yStart;
        const int x = p - (p / validWidth) * validWidth + xStart;

        const uchar center = image[y * width + x];

        const int xx = x + offset.x;
        const int yy = y + offset.y;

        const uchar value = image[yy * width + xx];

        int difference = (int)center - (int)value;

        if (difference < 0)
            difference = -difference;

        atomic_inc(&histogram[difference]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Write this work-group's partial histogram.
    __global uint* dst =
        partialHistograms +
        (offsetIndex * (uint)groupCount + subgroupIndex) * 256u;

    for (uint i = localId; i < 256; i += localSize)
        dst[i] = histogram[i];
}

)CLC";

    // -----------------------------------------------------------------------------
    // Select a reasonable number of groups per offset.
    //
    // The result is intentionally modest because every group has a 256-entry
    // local histogram and produces 1 KB of output.
    //
    // For a large image 16-32 groups per offset is generally enough.
    // -----------------------------------------------------------------------------

    static size_t chooseGroupCount(
        size_t pixelCount,
        size_t maxGroups)
    {
        if (pixelCount == 0)
            return 1;

        // Roughly one group per 4096 pixels.
        size_t groups =
            std::max<size_t>(
                1,
                pixelCount / 4096);

        groups =
            std::min(
                groups,
                maxGroups);

        return std::max<size_t>(1, groups);
    }

    // -----------------------------------------------------------------------------
    // Determine a legal work-group size.
    //
    // OpenCL implementations have different limits. 256 is convenient for the
    // 256-bin histogram, but we do not require it.
    // -----------------------------------------------------------------------------

    static size_t chooseLocalSize(
        cl_device_id device,
        cl_kernel kernel)
    {
        size_t deviceLimit = 1;
        size_t kernelLimit = 1;

        clGetDeviceInfo(
            device,
            CL_DEVICE_MAX_WORK_GROUP_SIZE,
            sizeof(deviceLimit),
            &deviceLimit,
            nullptr);

        clGetKernelWorkGroupInfo(
            kernel,
            device,
            CL_KERNEL_WORK_GROUP_SIZE,
            sizeof(kernelLimit),
            &kernelLimit,
            nullptr);

        size_t limit =
            std::min(
                deviceLimit,
                kernelLimit);

        if (limit >= 256)
            return 256;

        if (limit >= 128)
            return 128;

        if (limit >= 64)
            return 64;

        if (limit >= 32)
            return 32;

        if (limit >= 16)
            return 16;

        if (limit >= 8)
            return 8;

        if (limit >= 4)
            return 4;

        if (limit >= 2)
            return 2;

        return 1;
    }

    // -----------------------------------------------------------------------------
    // Main implementation
    // -----------------------------------------------------------------------------

    static Mat computeMaxDiffMatrixOpenCLImpl(
        const Mat& gray,
        int radius)
    {
        CV_Assert(!gray.empty());
        CV_Assert(gray.type() == CV_8U);
        CV_Assert(radius >= 1);

        const int K = radius * 2 + 1;

        // Same definition as the CPU implementation.
        const int border = 2;

        const int yStart = border;
        const int yEnd =
            gray.rows - border - radius;

        const int xStart =
            border + radius;

        const int xEnd =
            gray.cols - border - radius;

        if (yEnd <= yStart || xEnd <= xStart)
        {
            // Match the practical behavior expected from the CPU path for an
            // image that has no valid sample region.
            Mat result = Mat::zeros(
                K,
                K,
                CV_32F);

            return result;
        }

        const int validWidth =
            xEnd - xStart;

        const int validHeight =
            yEnd - yStart;

        const size_t pixelCount =
            static_cast<size_t>(validWidth) *
            static_cast<size_t>(validHeight);

        const std::vector<Offset> offsets =
            makeOffsets(radius);

        const int offsetCount =
            static_cast<int>(offsets.size());

        // CPU code has:
        //
        //     binsSize = (K - 1) * (radius + 1)
        //
        // which is identical to offsetCount.
        const int binsSize =
            (K - 1) * (radius + 1);

        CV_Assert(offsetCount == binsSize);

        // -------------------------------------------------------------------------
        // OpenCL setup
        // -------------------------------------------------------------------------

        Context cl;

        if (!chooseDevice(cl.platform, cl.device))
            throw std::runtime_error(
                "No OpenCL device available");

        cl_int err = CL_SUCCESS;

        cl.context =
            clCreateContext(
                nullptr,
                1,
                &cl.device,
                nullptr,
                nullptr,
                &err);

        checkCL(
            err,
            "clCreateContext");

        cl.queue =
            createCommandQueueCompat(
                cl.context,
                cl.device,
                cl.platform);

        if (!cl.queue)
        {
            throw std::runtime_error(
                "Unable to create OpenCL command queue");
        }

        // -------------------------------------------------------------------------
        // Program
        // -------------------------------------------------------------------------

        const char* source =
            kernelSource;

        const size_t sourceLength =
            std::strlen(source);

        cl.program =
            clCreateProgramWithSource(
                cl.context,
                1,
                &source,
                &sourceLength,
                &err);

        checkCL(
            err,
            "clCreateProgramWithSource");

        err =
            clBuildProgram(
                cl.program,
                1,
                &cl.device,
                nullptr,
                nullptr,
                nullptr);

        if (err != CL_SUCCESS)
        {
            const std::string log =
                getBuildLog(
                    cl.program,
                    cl.device);

            throw std::runtime_error(
                std::string("OpenCL kernel build failed: ") +
                clErrorString(err) +
                "\n" +
                log);
        }

        cl.kernel =
            clCreateKernel(
                cl.program,
                "buildPartialHistograms",
                &err);

        checkCL(
            err,
            "clCreateKernel");

        // -------------------------------------------------------------------------
        // Upload image.
        //
        // The CPU implementation accesses gray as a continuous logical image.
        // If the input is not continuous, make a compact copy.
        // -------------------------------------------------------------------------

        Mat compact;

        if (gray.isContinuous())
            compact = gray;
        else
            gray.copyTo(compact);

        const size_t imageBytes =
            compact.total() *
            sizeof(uchar);

        cl_mem imageBuffer =
            clCreateBuffer(
                cl.context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                imageBytes,
                compact.data,
                &err);

        checkCL(
            err,
            "clCreateBuffer(image)");

        // -------------------------------------------------------------------------
        // Upload offsets.
        // -------------------------------------------------------------------------

        std::vector<cl_int2> clOffsets(
            offsets.size());

        for (size_t i = 0; i < offsets.size(); ++i)
        {
            clOffsets[i].s[0] =
                static_cast<cl_int>(
                    offsets[i].dx);

            clOffsets[i].s[1] =
                static_cast<cl_int>(
                    offsets[i].dy);
        }

        cl_mem offsetBuffer =
            clCreateBuffer(
                cl.context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                clOffsets.size() *
                sizeof(cl_int2),
                clOffsets.data(),
                &err);

        checkCL(
            err,
            "clCreateBuffer(offsets)");

        // -------------------------------------------------------------------------
        // Work-group configuration.
        // -------------------------------------------------------------------------

        const size_t groupCount =
            chooseGroupCount(
                pixelCount,
                32);

        const size_t localSize =
            chooseLocalSize(
                cl.device,
                cl.kernel);

        const size_t totalGroups =
            static_cast<size_t>(offsetCount) *
            groupCount;

        const size_t globalSize =
            totalGroups *
            localSize;

        // Each work-group creates 256 uint32 histogram entries.
        //
        // Layout:
        //
        //     [offset][group][difference]
        //
        // where difference = 0..255.
        //
        const size_t partialHistogramEntries =
            totalGroups * 256;

        const size_t partialHistogramBytes =
            partialHistogramEntries *
            sizeof(uint32_t);

        cl_mem partialHistogramBuffer =
            clCreateBuffer(
                cl.context,
                CL_MEM_WRITE_ONLY,
                partialHistogramBytes,
                nullptr,
                &err);

        checkCL(
            err,
            "clCreateBuffer(partialHistograms)");

        // -------------------------------------------------------------------------
        // Kernel arguments
        // -------------------------------------------------------------------------

        const int width =
            compact.cols;

        const int height =
            compact.rows;

        const int groupCountInt =
            static_cast<int>(groupCount);

        checkCL(
            clSetKernelArg(
                cl.kernel,
                0,
                sizeof(cl_mem),
                &imageBuffer),
            "clSetKernelArg(image)");

        checkCL(
            clSetKernelArg(
                cl.kernel,
                1,
                sizeof(int),
                &width),
            "clSetKernelArg(width)");

        checkCL(
            clSetKernelArg(
                cl.kernel,
                2,
                sizeof(int),
                &height),
            "clSetKernelArg(height)");

        checkCL(
            clSetKernelArg(
                cl.kernel,
                3,
                sizeof(int),
                &xStart),
            "clSetKernelArg(xStart)");

        checkCL(
            clSetKernelArg(
                cl.kernel,
                4,
                sizeof(int),
                &yStart),
            "clSetKernelArg(yStart)");

        checkCL(
            clSetKernelArg(
                cl.kernel,
                5,
                sizeof(int),
                &validWidth),
            "clSetKernelArg(validWidth)");

        checkCL(
            clSetKernelArg(
                cl.kernel,
                6,
                sizeof(int),
                &validHeight),
            "clSetKernelArg(validHeight)");

        checkCL(
            clSetKernelArg(
                cl.kernel,
                7,
                sizeof(int),
                &radius),
            "clSetKernelArg(radius)");

        checkCL(
            clSetKernelArg(
                cl.kernel,
                8,
                sizeof(cl_mem),
                &offsetBuffer),
            "clSetKernelArg(offsets)");

        checkCL(
            clSetKernelArg(
                cl.kernel,
                9,
                sizeof(int),
                &offsetCount),
            "clSetKernelArg(offsetCount)");

        checkCL(
            clSetKernelArg(
                cl.kernel,
                10,
                sizeof(int),
                &groupCountInt),
            "clSetKernelArg(groupCount)");

        checkCL(
            clSetKernelArg(
                cl.kernel,
                11,
                sizeof(cl_mem),
                &partialHistogramBuffer),
            "clSetKernelArg(partialHistograms)");

        // -------------------------------------------------------------------------
        // Execute.
        // -------------------------------------------------------------------------

        err =
            clEnqueueNDRangeKernel(
                cl.queue,
                cl.kernel,
                1,
                nullptr,
                &globalSize,
                &localSize,
                0,
                nullptr,
                nullptr);

        checkCL(
            err,
            "clEnqueueNDRangeKernel");

        checkCL(
            clFinish(cl.queue),
            "clFinish");

        // -------------------------------------------------------------------------
        // Download partial histograms.
        // -------------------------------------------------------------------------

        std::vector<uint32_t> partialHistograms(
            partialHistogramEntries);

        checkCL(
            clEnqueueReadBuffer(
                cl.queue,
                partialHistogramBuffer,
                CL_TRUE,
                0,
                partialHistogramBytes,
                partialHistograms.data(),
                0,
                nullptr,
                nullptr),
            "clEnqueueReadBuffer");

        // Release temporary buffers before CPU processing.
        clReleaseMemObject(partialHistogramBuffer);
        clReleaseMemObject(offsetBuffer);
        clReleaseMemObject(imageBuffer);

        // -------------------------------------------------------------------------
        // Reduce partial histograms on CPU.
        //
        // Result layout is exactly:
        //
        //     bins[offset][difference]
        //
        // matching the existing CPU implementation.
        // -------------------------------------------------------------------------

        const int binsLength =
            binsSize * 256;

        std::vector<uint32_t> bins(
            static_cast<size_t>(binsLength),
            0);

        for (int offsetIndex = 0;
            offsetIndex < binsSize;
            ++offsetIndex)
        {
            uint32_t* destination =
                bins.data() +
                static_cast<size_t>(offsetIndex) * 256;

            for (size_t groupIndex = 0;
                groupIndex < groupCount;
                ++groupIndex)
            {
                const uint32_t* source =
                    partialHistograms.data() +
                    (
                        static_cast<size_t>(offsetIndex) *
                        groupCount +
                        groupIndex
                        ) * 256;

                for (int d = 0; d < 256; ++d)
                {
                    destination[d] +=
                        source[d];
                }
            }
        }

        // -------------------------------------------------------------------------
        // Exact same top-1% extraction as CPU implementation.
        // -------------------------------------------------------------------------

        const int nth =
            std::max(
                1,
                (gray.rows * gray.cols) / 100);

        std::vector<float> values;
        values.reserve(
            static_cast<size_t>(binsSize));

        for (int offsetIndex = 0;
            offsetIndex < binsSize;
            ++offsetIndex)
        {
            const uint32_t* histogram =
                bins.data() +
                static_cast<size_t>(offsetIndex) * 256;

            uint64_t sum = 0;
            uint64_t count = 0;

            // Same descending difference order.
            for (int d = 255;
                d >= 0 && count < static_cast<uint64_t>(nth);
                --d)
            {
                const uint32_t n =
                    histogram[d];

                if (n == 0)
                    continue;

                const uint64_t remaining =
                    static_cast<uint64_t>(nth) -
                    count;

                const uint64_t take =
                    std::min<uint64_t>(
                        n,
                        remaining);

                sum +=
                    static_cast<uint64_t>(d) *
                    take;

                count += take;
            }

            if (count != 0)
            {
                values.push_back(
                    static_cast<float>(
                        static_cast<double>(sum) /
                        static_cast<double>(count)));
            }
            else
            {
                values.push_back(0.0f);
            }
        }

        // -------------------------------------------------------------------------
        // Reconstruct M exactly as the CPU implementation.
        //
        // allValues =
        //
        //     reverse(values),
        //     0,
        //     values
        //
        // giving K*K entries.
        // -------------------------------------------------------------------------

        std::vector<float> allValues;

        allValues.reserve(
            static_cast<size_t>(K) *
            static_cast<size_t>(K));

        for (auto it = values.rbegin();
            it != values.rend();
            ++it)
        {
            allValues.push_back(*it);
        }

        allValues.push_back(0.0f);

        for (float value : values)
            allValues.push_back(value);

        CV_Assert(
            allValues.size() ==
            static_cast<size_t>(K) *
            static_cast<size_t>(K));

        Mat M(
            K,
            K,
            CV_32F,
            allValues.data());

        M = M.clone();

        // -------------------------------------------------------------------------
        // Same normalization as CPU implementation.
        // -------------------------------------------------------------------------

        double mn = 0.0;
        double mx = 0.0;

        minMaxLoc(
            M,
            &mn,
            &mx);

        Mat P =
            Mat::zeros(
                M.size(),
                M.type());

        if (mx - mn > 1e-12)
        {
            P =
                (mx - M) /
                static_cast<float>(mx - mn);
        }

        // -------------------------------------------------------------------------
        // Same unit-sum normalization used by the CPU path.
        //
        // This is written locally so this implementation does not depend on
        // MaxDiffOpenCL namespace visibility of the existing helper.
        // -------------------------------------------------------------------------

        const double total =
            sum(P)[0];

        if (std::abs(total) > 1e-12)
        {
            P /=
                static_cast<float>(total);
        }

        return P;
    }

    // -----------------------------------------------------------------------------
    // Public entry point
    //
    // Returns an empty Mat on OpenCL failure so the caller can choose to fall back
    // to the existing CPU implementation.
    //
    // If you prefer exceptions instead, remove the try/catch below.
    // -----------------------------------------------------------------------------

    Mat computeMaxDiffMatrixOpenCL(
        const Mat& gray,
        int radius)
    {
        try
        {
            return computeMaxDiffMatrixOpenCLImpl(
                gray,
                radius);
        }
        catch (const std::exception&)
        {
            // OpenCL is an optional acceleration path.
            //
            // The caller should normally fall back to:
            //
            //     computeMaxDiffMatrix(gray, radius)
            //
            // rather than making image processing fail merely because OpenCL is
            // unavailable or its driver has a problem.
            return Mat();
        }
    }

} // namespace MaxDiffOpenCL

//============================================================
// Helpers
//============================================================
static inline void normalizeToUnitSum(Mat& m)
{
    Scalar s = sum(m);
    if (std::abs((float)s[0]) > 1e-12f)
        m /= (float)s[0];
}

static inline void safeNormalizeMinMax(Mat& m)
{
    double mn, mx;
    minMaxLoc(m, &mn, &mx);
    const double d = mx - mn;
    if (d > 1e-12)
        m = (m - mn) / d;
    else
        m.setTo(0);
}

//============================================================
// Cached radial weighting for anomaly suppression
//============================================================
static Mat getRadialMidWeight(Size sz)
{
    static Size cachedSize;
    static Mat cached;

    if (cachedSize == sz && !cached.empty())
        return cached;

    cachedSize = sz;
    cached.create(sz, CV_32F);

    const int rows = sz.height;
    const int cols = sz.width;

    const float cx = cols * 0.5f;
    const float cy = rows * 0.5f;
    const float Rmax = std::min(cx, cy);

    for (int y = 0; y < rows; ++y)
    {
        float* row = cached.ptr<float>(y);
        const float dy = y - cy;

        for (int x = 0; x < cols; ++x)
        {
            const float dx = x - cx;
            const float r = std::sqrt(dx * dx + dy * dy) / Rmax;

            row[x] = std::exp(
                -0.5f * (r - 0.42f) * (r - 0.42f) / (0.18f * 0.18f));
        }
    }

    return cached;
}

//============================================================
// FFT shift
//============================================================
static void shiftPSF(const Mat& src, Mat& dst)
{
    dst.create(src.size(), src.type());

    const int cx = src.cols / 2;
    const int cy = src.rows / 2;

    src(Rect(cx, cy, src.cols - cx, src.rows - cy))
        .copyTo(dst(Rect(0, 0, src.cols - cx, src.rows - cy)));

    src(Rect(0, cy, cx, src.rows - cy))
        .copyTo(dst(Rect(src.cols - cx, 0, cx, src.rows - cy)));

    src(Rect(cx, 0, src.cols - cx, cy))
        .copyTo(dst(Rect(0, src.rows - cy, src.cols - cx, cy)));

    src(Rect(0, 0, cx, cy))
        .copyTo(dst(Rect(src.cols - cx, src.rows - cy, cx, cy)));
}

//typedef std::array<uint32_t, 256> TopKMean;

//============================================================
// Single-pass industrial computeMaxDiffMatrix
//============================================================
template<int radius>
Mat doComputeMaxDiffMatrix(const Mat& gray)
{
    CV_Assert(gray.type() == CV_8U);

    const int K = radius * 2 + 1;
    const int border = 2;

    const int binsSize = (K - 1) * (radius + 1);
    static_assert(binsSize * 2 + 1 == K * K);

    const int yStart = border;
    const int yEnd = gray.rows - border - radius;
    const int xStart = border + radius;
    const int xEnd = gray.cols - border - radius;

    const int binsLen = binsSize * 256;

    // Глобальный bins (результат после редукции)
    std::vector<uint32_t> bins(binsLen, 0);

    // Параллельный проход по y
#pragma omp parallel
    {
        std::array<const uchar*, radius + 1> nbrRows{};
        std::vector<uint32_t> localBins(binsLen, 0);

#pragma omp for nowait
        for (int y = yStart; y < yEnd; ++y)
        {
            for (int j = 0; j <= radius; ++j)
                nbrRows[j] = gray.ptr<uchar>(y + j);

            const uchar* centerRow = gray.ptr<uchar>(y);

            for (int x = xStart; x < xEnd; ++x)
            {
                const uchar c = centerRow[x];

                int dx = 1;
                uint32_t* idx1 = localBins.data();

                for (int dy = 0; dy <= radius; ++dy)
                {
                    const uchar* rowPos = nbrRows[dy];

                    for (; dx <= radius; ++dx, idx1 += 256)
                    {
                        const int d = std::abs(c - rowPos[x + dx]);
                        ++idx1[d];
                    }

                    dx = -radius;
                }

#ifndef NDEBUG
                if (idx1 - localBins.data() != binsLen)
                    CV_Error(Error::StsInternal, "Index mismatch in bins");
#endif
            }
        }

        // Редукция локальных bins в глобальный
#pragma omp critical
        {
            for (int i = 0; i < binsLen; ++i)
                bins[i] += localBins[i];
        }
    }

    std::vector<float> values;
    values.reserve(binsSize);

    const int nth = std::max(1, (gray.rows * gray.cols) / 100);

    for (int j = 0; j < binsSize; ++j)
    {
        uint32_t* bin = bins.data() + j * 256;
        float sum = 0.0f;
        int count = 0;

        for (int i = 255; i >= 0; --i)
        {
            const uint32_t v = bin[i];
            if (count + v >= nth)
            {
                sum += (nth - count) * i;
                count = nth;
                break;
            }
            sum += v * i;
            count += v;
        }

        values.push_back(count > 0 ? sum / count : 0.0f);
    }

    std::vector<float> allValues(values.rbegin(), values.rend());
    allValues.push_back(0.0f);
    allValues.insert(allValues.end(), values.begin(), values.end());

    if (allValues.size() != K * K)
        CV_Error(Error::StsInternal, "Total values count mismatch");

    Mat M(K, K, CV_32F, allValues.data());

    double mn, mx;
    minMaxLoc(M, &mn, &mx);

    Mat P;
    if (mx - mn > 1e-12)
        P = (mx - M) / (mx - mn);
    else
        P = Mat::zeros(M.size(), CV_32F);

    normalizeToUnitSum(P);
    return P;
}

Mat computeMaxDiffMatrix(const Mat& gray, int radius)
{
    switch (radius)
    {
    case 1: return doComputeMaxDiffMatrix<1>(gray);
    case 2: return doComputeMaxDiffMatrix<2>(gray);
    case 3: return doComputeMaxDiffMatrix<3>(gray);
    case 4: return doComputeMaxDiffMatrix<4>(gray);
    case 5: return doComputeMaxDiffMatrix<5>(gray);
    case 6: return doComputeMaxDiffMatrix<6>(gray);
    case 7: return doComputeMaxDiffMatrix<7>(gray);
    case 8: return doComputeMaxDiffMatrix<8>(gray);
    case 9: return doComputeMaxDiffMatrix<9>(gray);
    case 10: return doComputeMaxDiffMatrix<10>(gray);
    case 11: return doComputeMaxDiffMatrix<11>(gray);
    case 12: return doComputeMaxDiffMatrix<12>(gray);
    case 13: return doComputeMaxDiffMatrix<13>(gray);
    case 14: return doComputeMaxDiffMatrix<14>(gray);
    case 15: return doComputeMaxDiffMatrix<15>(gray);
    default:
        CV_Error(Error::StsBadArg, "Unsupported radius");
        return Mat();
    }
}



//============================================================
Mat clipPSFByHeap(const Mat& P, float fraction = 0.8f)
{
    std::vector<float> vals;
    vals.reserve(P.total());

    for (int y = 0; y < P.rows; ++y)
    {
        const float* row = P.ptr<float>(y);
        vals.insert(vals.end(), row, row + P.cols);
    }

    const int k = std::max(1, int(vals.size() * fraction));
    std::nth_element(vals.begin(), vals.begin() + k - 1, vals.end());
    const float threshold = vals[k - 1];

    Mat out = P.clone();

    for (int y = 0; y < out.rows; ++y)
    {
        float* row = out.ptr<float>(y);
        for (int x = 0; x < out.cols; ++x)
            row[x] = std::max(0.0f, row[x] - threshold);
    }

    return out;
}

//============================================================
Mat cropPSFToActiveRegionAndFixOdd(const Mat& psf)
{
    int top = 0, bottom = psf.rows - 1;
    int left = 0, right = psf.cols - 1;

    while (top <= bottom)
    {
        bool nz = false;
        const float* r1 = psf.ptr<float>(top);
        const float* r2 = psf.ptr<float>(bottom);

        for (int x = 0; x < psf.cols; ++x)
            if (r1[x] != 0.0f || r2[x] != 0.0f) { nz = true; break; }

        if (nz) break;
        ++top; --bottom;
    }

    while (left <= right)
    {
        bool nz = false;
        for (int y = top; y <= bottom; ++y)
        {
            const float* row = psf.ptr<float>(y);
            if (row[left] != 0.0f || row[right] != 0.0f) { nz = true; break; }
        }

        if (nz) break;
        ++left; --right;
    }

    if (top > bottom || left > right)
        return psf.clone();

    return psf(Rect(left, top, right - left + 1, bottom - top + 1)).clone();
}

//============================================================
Mat computeCorrelationFFT(const Mat& gray, int radius)
{
    Mat f32;
    gray.convertTo(f32, CV_32F);
    f32 -= mean(f32)[0];

    Mat gx, gy;
    Sobel(f32, gx, CV_32F, 1, 0, 3);
    Sobel(f32, gy, CV_32F, 0, 1, 3);

    magnitude(gx, gy, f32);
    threshold(f32, f32, 10.0f, 0.0f, THRESH_TOZERO);

    Mat F;
    {
        Mat planes[] = { f32, Mat::zeros(f32.size(), CV_32F) };
        merge(planes, 2, F);
        dft(F, F, DFT_COMPLEX_OUTPUT);
    }

    Mat Fc;
    mulSpectrums(F, F, Fc, 0, true);

    Mat C;
    idft(Fc, C, DFT_REAL_OUTPUT | DFT_SCALE);

    Mat shifted;
    shiftPSF(C, shifted);

    const int cx = shifted.cols / 2;
    const int cy = shifted.rows / 2;

    Mat cropped = shifted(Rect(cx - radius, cy - radius, 2 * radius + 1, 2 * radius + 1)).clone();

    safeNormalizeMinMax(cropped);
    normalizeToUnitSum(cropped);

    return cropped;
}

//============================================================
Mat buildPSFFromM(const Mat& M)
{
    Mat P = clipPSFByHeap(M);
    P = cropPSFToActiveRegionAndFixOdd(P);
    normalizeToUnitSum(P);
    return P;
}

//============================================================
Mat buildInverseFilterFromPSF(const Mat& psfSmall, Size imgSize, float K = 0.01f)
{
    Mat psfPadded(imgSize, CV_32F, Scalar(0));

    const int x0 = (imgSize.width - psfSmall.cols) / 2;
    const int y0 = (imgSize.height - psfSmall.rows) / 2;

    psfSmall.copyTo(psfPadded(Rect(x0, y0, psfSmall.cols, psfSmall.rows)));

    Mat psfShifted;
    shiftPSF(psfPadded, psfShifted);

    Mat H;
    Mat planesH[] = { psfShifted, Mat::zeros(imgSize, CV_32F) };
    merge(planesH, 2, H);
    dft(H, H, DFT_COMPLEX_OUTPUT);

    Mat planes[2];
    split(H, planes);

    Mat& Re = planes[0];
    Mat& Im = planes[1];

    Mat mag2 = Re.mul(Re);
    mag2 += Im.mul(Im);

    const float adaptiveK = K * (float)(mean(mag2)[0] + 1e-8f);
    Mat denom = mag2 + adaptiveK;

    Re /= denom;
    Im /= denom;
    Im = -Im;

    Mat G;
    Mat outp[] = { Re, Im };
    merge(outp, 2, G);
    return G;
}

//============================================================
static void DeQuantization(Mat* planes)
{
    Mat& Re = planes[0];
    Mat& Im = planes[1];

    const float dcRe = Re.at<float>(0, 0);
    const float dcIm = Im.at<float>(0, 0);

    const float sigma2_axis =
        (1.0f / 12.0f) * static_cast<float>(Re.total()) * 0.5f;

    for (int y = 0; y < Re.rows; ++y)
    {
        float* re = Re.ptr<float>(y);
        float* im = Im.ptr<float>(y);

        for (int x = 0; x < Re.cols; ++x)
        {
            const float mag2 = re[x] * re[x] + im[x] * im[x] + 1e-12f;
            float alpha = 1.0f - sigma2_axis / mag2;
            if (alpha < 0.0f) alpha = 0.0f;

            re[x] *= alpha;
            im[x] *= alpha;
        }
    }

    Re.at<float>(0, 0) = dcRe;
    Im.at<float>(0, 0) = dcIm;
}

//============================================================
static void AnomalySuppression(Mat* cpl)
{
    Mat& Re = cpl[0];
    Mat& Im = cpl[1];

    Mat work;
    magnitude(Re, Im, work);
    work += 1.0f;
    log(work, work);

    Mat baseline;
    GaussianBlur(work, baseline, Size(0, 0), 18.0, 18.0, BORDER_REFLECT);

    work -= baseline;

    Scalar mu, sd;
    meanStdDev(work, mu, sd);

    const float meanv = (float)mu[0];
    const float stdv = std::max(0.001f, (float)sd[0]);

    const int rows = work.rows;
    const int cols = work.cols;

    for (int y = 0; y < rows; ++y)
    {
        float* wp = work.ptr<float>(y);

        for (int x = 0; x < cols; ++x)
        {
            const float z = (wp[x] - meanv) / stdv;
            wp[x] = (z < 1.5f) ? 0.0f : std::min(1.0f, (z - 1.5f) / 2.5f);
        }
    }

    GaussianBlur(work, work, Size(0, 0), 3.5, 3.5, BORDER_REFLECT);

    const Mat radial = getRadialMidWeight(work.size());

    for (int y = 0; y < rows; ++y)
    {
        float* wp = work.ptr<float>(y);
        const float* rp = radial.ptr<float>(y);

        for (int x = 0; x < cols; ++x)
            wp[x] = 1.0f - 0.30f * wp[x] * rp[x];
    }

    GaussianBlur(work, work, Size(0, 0), 2.0, 2.0, BORDER_REFLECT);

    for (int y = 0; y < rows; ++y)
    {
        const float* mp = work.ptr<float>(y);
        float* re = Re.ptr<float>(y);
        float* im = Im.ptr<float>(y);

        for (int x = 0; x < cols; ++x)
        {
            const float m = mp[x];
            re[x] *= m;
            im[x] *= m;
        }
    }
}

//============================================================
Mat applyFilterDFT(const Mat& F, const Mat& G)
{
    Mat Y;
    mulSpectrums(F, G, Y, 0);

    Mat out32;
    idft(Y, out32, DFT_REAL_OUTPUT | DFT_SCALE);

    Mat out8;
    out32.convertTo(out8, CV_8U);
    return out8;
}

//============================================================
Mat deblurChannel(const Mat& gray)
{
    Mat Y;
    gray.convertTo(Y, CV_32F);

    const int radius = 15;
    //*
    Mat M = MaxDiffOpenCL::computeMaxDiffMatrixOpenCL(gray, radius);

    if (M.empty())
        M = computeMaxDiffMatrix(gray, radius);

    //M += computeCorrelationFFT(Y, radius);
    //*/
    //Mat M = computeMaxDiffMatrix(gray, radius) + computeCorrelationFFT(Y, radius);
    Mat psf = buildPSFFromM(M);
    Mat G = buildInverseFilterFromPSF(psf, Y.size(), 0.1f);

    Mat F;
    {
        Mat planes0[] = { Y, Mat::zeros(Y.size(), CV_32F) };
        merge(planes0, 2, F);
        dft(F, F, DFT_COMPLEX_OUTPUT);
    }

    Mat planes[2];
    split(F, planes);

    DeQuantization(planes);
    AnomalySuppression(planes);

    merge(planes, 2, F);

    return applyFilterDFT(F, G);
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <image_path> <output_path>\n";
        return -1;
    }

    Mat img = imread(argv[1]);
    if (img.empty()) return -1;

    // BGR → YCrCb
    Mat ycrcb;
    cvtColor(img, ycrcb, COLOR_BGR2YCrCb);
    std::vector<Mat> ch;
    split(ycrcb, ch);

    Mat Cr = ch[1];
    Mat Cb = ch[2];

    auto start = std::chrono::high_resolution_clock::now();
    ch[0] = deblurChannel(ch[0]);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Processing time: " << elapsed.count() << " seconds\n";

    merge(ch, ycrcb);

    Mat restoredBGR;
    cvtColor(ycrcb, restoredBGR, COLOR_YCrCb2BGR);

    imwrite(argv[2], restoredBGR);
    if (argc > 3)
        imwrite(argv[3], ch[0]);

    return 0;
}
