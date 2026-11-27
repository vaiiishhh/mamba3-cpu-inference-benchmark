#include <iostream>
#include <cstring>
#include <thread>
#include "platform.h"
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif
int main() try {
    char brand[49]{};
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int info[4]; __cpuid(info, int(0x80000000));
    if (unsigned(info[0]) >= 0x80000004)
        for (int i = 0; i < 3; ++i) { __cpuid(info, int(0x80000002u + i)); std::memcpy(brand + i*16, info, 16); }
#elif defined(__x86_64__) || defined(__i386__)
    if (__get_cpuid_max(0x80000000, nullptr) >= 0x80000004)
        for (unsigned i = 0; i < 3; ++i) {
            unsigned a,b,c,d; __cpuid(0x80000002+i,a,b,c,d);
            unsigned values[] = {a,b,c,d}; std::memcpy(brand + i*16, values, 16);
        }
#elif defined(__APPLE__)
    size_t length = sizeof(brand); sysctlbyname("machdep.cpu.brand_string", brand, &length, nullptr, 0);
#endif
    std::cout << "cpu=" << (brand[0] ? brand : "ARM64 CPU") << "\n";
    std::cout << "logical_cpus=" << std::thread::hardware_concurrency() << "\n";
    std::cout << "free_ram_bytes=" << free_ram() << "\n";
#ifdef _WIN32
    MEMORYSTATUSEX m{}; m.dwLength=sizeof(m); GlobalMemoryStatusEx(&m);
    SYSTEM_POWER_STATUS power{}; GetSystemPowerStatus(&power);
    auto cores = performance_cores();
    std::cout << "ram_bytes=" << m.ullTotalPhys << "\nac_line_status=" << int(power.ACLineStatus)
              << "\naffinity=group-0-performance-cores\nworker_count=" << cores.size() << "\np_core_masks=";
    for (auto mask : cores) std::cout << mask << ',';
    std::cout << "\ncpu_indices=";
    for (auto mask : cores) { int cpu = 0; while ((mask >>= 1)) ++cpu; std::cout << cpu << ','; }
    std::cout << "\n";
    DWORD bytes=0; GetLogicalProcessorInformationEx(RelationAll,nullptr,&bytes);
    std::vector<char> data(bytes);
    if (!GetLogicalProcessorInformationEx(RelationAll,
        reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(data.data()),&bytes))
        throw std::runtime_error("Cannot query caches");
    for (DWORD offset=0; offset<bytes;) {
        auto *i=reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(data.data()+offset);
        if (i->Relationship==RelationCache)
            std::cout << "cache=" << unsigned(i->Cache.Level) << ',' << i->Cache.CacheSize << ',' << unsigned(i->Cache.Type) << "\n";
        offset+=i->Size;
    }
#elif defined(__APPLE__)
    auto count = sysctl_number("hw.perflevel0.physicalcpu");
    if (!count) count = sysctl_number("hw.physicalcpu");
    std::cout << "ram_bytes=" << sysctl_number("hw.memsize") << "\nworker_count=" << count
              << "\naffinity=unavailable (macOS scheduler placement)\ncpu_indices=\n";
    auto l3 = sysctl_number("hw.l3cachesize");
    if (l3) std::cout << "cache=3," << l3 << ",0\n";
#else
    auto cores = linux_cores();
    std::cout << "worker_count=" << cores.size() << "\naffinity=allowed-physical-cores\ncpu_indices=";
    for (int cpu : cores) std::cout << cpu << ',';
    std::cout << "\n";
    // Deduplicate shared last-level caches across selected CPUs.
    std::set<std::string> caches;
    for (int cpu : cores) for (int index = 0; index < 8; ++index) {
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cache/index" + std::to_string(index) + "/";
        int level = 0; std::string size, shared;
        std::ifstream(base + "level") >> level;
        std::ifstream(base + "size") >> size;
        std::ifstream(base + "shared_cpu_list") >> shared;
        if (level == 3 && !size.empty() && caches.insert(shared).second) {
            auto bytes = std::stoull(size);
            if (size.back() == 'K') bytes *= 1024;
            if (size.back() == 'M') bytes *= 1024 * 1024;
            std::cout << "cache=3," << bytes << ",0\n";
        }
    }
#endif
    return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
