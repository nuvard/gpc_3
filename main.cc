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
    printf("initial size is: %i \n", n);
    cl::Kernel kernel_map(opencl.program, "map_positive");
    cl::Kernel kernel_scan(opencl.program, "scan_inclusive");
    cl::Kernel kernel_sum(opencl.program, "add_chunk_sum");
    cl::Kernel kernel_pack(opencl.program, "pack");

    int group_size = 64;
    int iter_num = 0;

    std::vector<cl::Buffer> scans;
    std::vector<int> scan_sizes;

    auto t0 = clock_type::now();
    filter(input, expected_result, [] (float x) { return x > 0; }); // filter positive numbers
    auto t1 = clock_type::now();
    cl::Buffer d_input(opencl.queue, begin(input), end(input), true);
    cl::Buffer d_mask(opencl.context, CL_MEM_READ_WRITE, (n + group_size)*sizeof(int)); // n+group_size чтобы сработал первый цикл сканса
    auto t2 = clock_type::now();
    //первое - просто строим маску позитивности
    kernel_map.setArg(0, d_input);
    kernel_map.setArg(1, d_mask);

    opencl.queue.enqueueNDRangeKernel(kernel_map,cl::NullRange,cl::NDRange(n),cl::NullRange);
    scans.push_back(d_mask);
    //для маски делаем скан (как во второй лабе - копипаста)
    for (int scan_size = n; scan_size > 1; scan_size = (scan_size + group_size - 1) / group_size) {
        scan_sizes.push_back(scan_size);
        iter_num++;
        scans.push_back(cl::Buffer(opencl.context, CL_MEM_READ_WRITE, (scan_size+group_size)*sizeof(float)));
        kernel_scan.setArg(0, scans[iter_num-1]);
        kernel_scan.setArg(1, cl::Local(group_size*sizeof(float)));
        kernel_scan.setArg(2, scans[iter_num]);
        kernel_scan.setArg(3, scan_size);
        kernel_scan.setArg(4, group_size);
        opencl.queue.enqueueNDRangeKernel(
                kernel_scan,
                cl::NullRange,
                cl::NDRange(((scan_size + group_size - 1)/ group_size) * group_size),
                cl::NDRange(group_size));
        opencl.queue.flush();
    }

    for (int i = iter_num - 1; i > 0; i -= 1) {
        kernel_sum.setArg(0, scans[i-1]);
        kernel_sum.setArg(1, scans[i]);
        kernel_sum.setArg(2, scan_sizes[i-1]);
        kernel_sum.setArg(3, group_size);
        opencl.queue.enqueueNDRangeKernel(
                kernel_sum,
                cl::NullRange,
                cl::NDRange(((scan_sizes[i-1] + group_size - 1) / group_size) * group_size),
                cl::NDRange(group_size)
        );
    }

    //собираем результаты
    cl::Buffer d_result(opencl.context, CL_MEM_READ_WRITE, (n)*sizeof(int));
    std::vector<int> final_masks(n);
    kernel_pack.setArg(0, d_input);
    kernel_pack.setArg(1, scans[0]);
    kernel_pack.setArg(2, d_result);

    opencl.queue.enqueueNDRangeKernel(kernel_pack,cl::NullRange, cl::NDRange(n-1), cl::NullRange);
    opencl.queue.flush();

    auto t3 = clock_type::now();
    opencl.queue.enqueueReadBuffer(scans[0], true, 0, final_masks.size()*sizeof(int), final_masks.data());

    int size = final_masks.back(); //последний элемент - должен давать количестыо чисел больше нуля
    print("After masking: %i\n", size);
    result.resize(size);
    opencl.queue.enqueueReadBuffer(d_result, true, 0, n*sizeof(float), result.data());
    opencl.queue.flush();
    auto t4 = clock_type::now();
    verify_vector(expected_result, result);
    print("filter", {t1-t0,t4-t1,t2-t1,t3-t2,t4-t3});
}

const std::string src = R"(
kernel void add_chunk_sum(global float * a,
                          global float * chunk_sums,
                          int current_size,
                          int group_size
                          ) {
  int global_id = get_global_id(0);
  int group_id = get_group_id(0);
  if (global_id >= group_size && global_id < current_size){ //необходимо оставить первую группу как есть
      a[global_id] += chunk_sums[group_id-1];
  }
}

kernel void map_positive(
    global float * a,
    global int * result
    ) {
    int i = get_global_id(0);
    result[i] = a[i] > 0 ? 1 : 0;
}

kernel void pack(
    global float * a,
    global int * mask,
    global float * result
    ) {
    int i = get_global_id(0);
    if (mask[i] < mask[i+1]) {
        result[mask[i]] = a[i+1];
    }
}

scan_inclusive(global int * data,
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
