#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl2.hpp>
#endif

#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include "filter.hh"
#include "linear-algebra.hh"
#include "reduce-scan.hh"

using clock_type = std::chrono::high_resolution_clock;
using duration = clock_type::duration;
using time_point = clock_type::time_point;

double bandwidth(int n, time_point t0, time_point t1) {
    using namespace std::chrono;
    const auto dt = duration_cast<microseconds>(t1-t0).count();
    if (dt == 0) { return 0; }
    return ((n+n+n)*sizeof(float)*1e-9)/(dt*1e-6);
}

void print(const char* name, std::array<duration,5> dt) {
    using namespace std::chrono;
    std::cout << std::setw(19) << name;
    for (size_t i=0; i<5; ++i) {
        std::stringstream tmp;
        tmp << duration_cast<microseconds>(dt[i]).count() << "us";
        std::cout << std::setw(20) << tmp.str();
    }
    std::cout << '\n';
}

void print_column_names() {
    std::cout << std::setw(19) << "function";
    std::cout << std::setw(20) << "OpenMP";
    std::cout << std::setw(20) << "OpenCL total";
    std::cout << std::setw(20) << "OpenCL copy-in";
    std::cout << std::setw(20) << "OpenCL kernel";
    std::cout << std::setw(20) << "OpenCL copy-out";
    std::cout << '\n';
}

struct OpenCL {
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    cl::CommandQueue queue;
};

void profile_filter(int n, OpenCL& opencl) {
    auto input = random_std_vector<float>(n);
    std::vector<float> result, expected_result;
    result.reserve(n);
    cl::Kernel scan(opencl.program, "scan_inclusive");
    cl::Kernel scan_final(opencl.program, "scan_final");
    cl::Kernel map(opencl.program, "map_more_zero");
    cl::Kernel scatter(opencl.program, "scatter");
    std::vector<cl::Buffer> buffers;
    std::vector<int> buffer_sizes;
    auto t0 = clock_type::now();
    filter(input, expected_result, [] (float x) { return x > 0; }); // filter positive numbers
    auto t1 = clock_type::now();
    int buffer_size = 16;
    int arg = 0;
    cl::Buffer inputbuffer(opencl.queue, begin(input), end(input), true);
    cl::Buffer mask(opencl.context, CL_MEM_READ_WRITE, (n + buffer_size)*sizeof(int));
    auto t2 = clock_type::now();
    map.setArg(0, inputbuffer);
    map.setArg(1, mask);
    opencl.queue.enqueueNDRangeKernel(
            map,
            cl::NullRange,
            cl::NDRange(n),
            cl::NullRange
    );
    buffers.push_back(mask);

    for (int size = n; size > 1; size = (size+ buffer_size - 1)/ buffer_size) {
        scan.setArg(0, buffers[arg]);
        arg++;
        buffers.emplace_back(opencl.context, CL_MEM_READ_WRITE, (size + buffer_size)*sizeof(int));
        buffer_sizes.push_back(size);
        scan.setArg(2, buffers[arg]);
        //if (size < buffer_size) buffer_size = size;
        scan.setArg(1, cl::Local(buffer_size*sizeof(int)));
        scan.setArg(3, size);
        scan.setArg(4, buffer_size);
        opencl.queue.enqueueNDRangeKernel(
                scan,
                cl::NullRange,
                cl::NDRange(((size+ buffer_size - 1)/ buffer_size) * buffer_size),
                cl::NDRange(buffer_size) );
        opencl.queue.flush();
    }

    for (int i = arg-1; i >= 1; i--) {
        scan_final.setArg(0, buffers[i-1]);
        scan_final.setArg(1, buffers[i]);
        scan_final.setArg(2, buffer_sizes[i-1]);
        scan_final.setArg(3, buffer_size);
        opencl.queue.enqueueNDRangeKernel(
                scan_final,
                cl::NullRange,
                cl::NDRange(((buffer_sizes[i-1]+ buffer_size - 1)/ buffer_size)*buffer_size),
                cl::NDRange(buffer_size)
        );
    }

    cl::Buffer resultbuffer(opencl.context, CL_MEM_READ_WRITE, (n)*sizeof(int));
    std::vector<int> fin_masks(n);
    scatter.setArg(0, inputbuffer);
    scatter.setArg(1, buffers[0]);
    scatter.setArg(2, resultbuffer);

    opencl.queue.enqueueNDRangeKernel(
            scatter,
            cl::NullRange,
            cl::NDRange(n-1),
            cl::NullRange
    );
    opencl.queue.flush();

    auto t3 = clock_type::now();
    opencl.queue.enqueueReadBuffer(buffers[0], true, 0, fin_masks.size()*sizeof(int), fin_masks.data());
    int size = fin_masks.back();
    result.resize(size);
    opencl.queue.enqueueReadBuffer(resultbuffer, true, 0, n*sizeof(float), result.data());
    opencl.queue.flush();

    auto t4 = clock_type::now();
    // TODO Implement OpenCL version! See profile_vector_times_vector for an example.
    // TODO Uncomment the following line!
    verify_vector(expected_result, result);
    print("filter", {t1-t0,t4-t1,t2-t1,t3-t2,t4-t3});
}

const std::string src = R"(

kernel void add_chunk_sum(
    global int* data,
    global int* scan,
    int size,
    int buffer_size
) {
    int gid = get_global_id(0);
    int grid = get_group_id(0);

    if (gid >= buffer_size && gid < size){
        data[gid] += scan[grid-1];
    }
}

kernel void map_positive(
    global float* data,
    global int* result
) {
    int i = get_global_id(0);
    result[i] = data[i] > 0 ? 1 : 0;
}

kernel void pack(
    global float* a,
    global int* mask,
    global float* result
) {
    int i = get_global_id(0);
    if (mask[i] < mask[i+1]) {
        result[mask[i]] = a[i+1];
    }
}

kernel void scan_inclusive(global int * data,
                   local int * buffer,
                   global int * result,
                   int size,
                   int buffer_size) {
    int local_id = get_local_id(0); // номер потока в группе
    int group_id = get_group_id(0);
    int global_id = get_global_id(0);
    if (global_id < size)
        buffer[local_id] = data[global_id];
    else
        buffer[local_id] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int offset=1; offset<buffer_size; offset *= 2) {
        int sum = 0;
        if (global_id < size && local_id >= offset) {
            sum += buffer[local_id - offset];
            buffer[local_id] += sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (global_id < size) {
        data[global_id] = buffer[local_id];
    }

    if (local_id == 0) {
        //printf("%.3f val\n %d group_id\n%d size\n%d buffer_size\n\n", buffer[0], group_id, size, buffer_size);
        result[group_id] = buffer[buffer_size-1];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}
)";

void opencl_main(OpenCL& opencl) {
    using namespace std::chrono;
    print_column_names();
    profile_filter(1024*1024, opencl);
}

int main() {
    try {
        // find OpenCL platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            std::cerr << "Unable to find OpenCL platforms\n";
            return 1;
        }
        cl::Platform platform = platforms[0];
        std::clog << "Platform name: " << platform.getInfo<CL_PLATFORM_NAME>() << '\n';
        // create context
        cl_context_properties properties[] =
            { CL_CONTEXT_PLATFORM, (cl_context_properties)platform(), 0};
        cl::Context context(CL_DEVICE_TYPE_GPU, properties);
        // get all devices associated with the context
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        cl::Device device = devices[0];
        std::clog << "Device name: " << device.getInfo<CL_DEVICE_NAME>() << '\n';
        cl::Program program(context, src);
        // compile the programme
        try {
            program.build(devices);
        } catch (const cl::Error& err) {
            for (const auto& device : devices) {
                std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                std::cerr << log;
            }
            throw;
        }
        cl::CommandQueue queue(context, device);
        OpenCL opencl{platform, device, context, program, queue};
        opencl_main(opencl);
    } catch (const cl::Error& err) {
        std::cerr << "OpenCL error in " << err.what() << '(' << err.err() << ")\n";
        std::cerr << "Search cl.h file for error code (" << err.err()
            << ") to understand what it means:\n";
        std::cerr << "https://github.com/KhronosGroup/OpenCL-Headers/blob/master/CL/cl.h\n";
        return 1;
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        return 1;
    }
    return 0;
}
