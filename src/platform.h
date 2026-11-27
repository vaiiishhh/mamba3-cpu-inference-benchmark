#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
inline unsigned efficiency_class(const PROCESSOR_RELATIONSHIP &p) {
    // Windows ABI puts EfficiencyClass at byte 1. Older MinGW headers name
    // that byte Reserved[0], while Windows 11 fills it using the current ABI.
    return reinterpret_cast<const BYTE *>(&p)[1];
}
inline std::vector<ULONG_PTR> performance_cores() {
    DWORD size = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &size);
    std::vector<char> data(size);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(data.data()), &size))
        throw std::runtime_error("Cannot query CPU topology");
    std::vector<std::pair<unsigned, ULONG_PTR>> cores;
    unsigned best = 0;
    for (DWORD offset = 0; offset < size;) {
        auto *info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(data.data() + offset);
        auto &p = info->Processor;
        if (p.GroupCount != 1 || p.GroupMask[0].Group != 0) {
            offset += info->Size;
            continue; // Restrict to group 0; never fabricate CPU indices for other groups.
        }
        best = std::max(best, efficiency_class(p));
        auto mask = p.GroupMask[0].Mask;
        cores.push_back({efficiency_class(p), mask & (~mask + 1)});
        offset += info->Size;
    }
    if (cores.empty()) throw std::runtime_error("No available cores in Windows processor group 0");
    std::vector<ULONG_PTR> result;
    for (auto core : cores) if (core.first == best) result.push_back(core.second);
    return result;
}
inline void pin_performance_cores(bool enabled = true, size_t count = 0) {
    if (!enabled) return;
    ULONG_PTR mask = 0;
    auto cores = performance_cores();
    if (count > cores.size()) throw std::runtime_error("Requested more performance cores than available");
    if (count) cores.resize(count);
    for (auto core : cores) mask |= core;
    if (!SetProcessAffinityMask(GetCurrentProcess(), mask))
        throw std::runtime_error("Cannot set performance-core affinity");
}
inline uint64_t peak_rss() {
    PROCESS_MEMORY_COUNTERS info{};
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info)))
        throw std::runtime_error("Cannot read peak working set");
    return info.PeakWorkingSetSize;
}
inline uint64_t free_ram() {
    MEMORYSTATUSEX info{};
    info.dwLength = sizeof(info);
    GlobalMemoryStatusEx(&info);
    return info.ullAvailPhys;
}
#else
#include <sys/resource.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
inline uint64_t sysctl_number(const char *name) {
    uint64_t value = 0; size_t size = sizeof(value);
    if (sysctlbyname(name, &value, &size, nullptr, 0) != 0) return 0;
    return value;
}
inline void pin_performance_cores(bool = true, size_t = 0) {}
inline uint64_t peak_rss() {
    rusage r{};
    if (getrusage(RUSAGE_SELF, &r)) throw std::runtime_error("Cannot read peak RSS");
    return uint64_t(r.ru_maxrss); // macOS reports bytes, Linux reports KiB.
}
inline uint64_t free_ram() {
    vm_statistics64_data_t vm{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          reinterpret_cast<host_info64_t>(&vm), &count) != KERN_SUCCESS)
        throw std::runtime_error("Cannot read available RAM");
    return (uint64_t(vm.free_count) + vm.inactive_count) * uint64_t(sysconf(_SC_PAGESIZE));
}
#else
#include <fstream>
#include <set>
#include <string>
#include <sched.h>
inline std::vector<int> linux_cores() {
    cpu_set_t allowed; CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed))
        throw std::runtime_error("Cannot query allowed CPU affinity");
    std::set<std::pair<int, int>> seen;
    std::vector<int> result;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) if (CPU_ISSET(cpu, &allowed)) {
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
        int core = cpu, package = 0;
        std::ifstream(base + "core_id") >> core;
        std::ifstream(base + "physical_package_id") >> package;
        if (seen.insert({package, core}).second) result.push_back(cpu);
    }
    return result;
}
inline void pin_performance_cores(bool enabled = true, size_t count = 0) {
    if (!enabled) return;
    auto cores = linux_cores();
    if (count > cores.size()) throw std::runtime_error("More workers than allowed physical cores");
    if (count) cores.resize(count);
    cpu_set_t selected; CPU_ZERO(&selected);
    for (int cpu : cores) CPU_SET(cpu, &selected);
    if (sched_setaffinity(0, sizeof(selected), &selected))
        throw std::runtime_error("Cannot set CPU affinity");
}
inline uint64_t peak_rss() {
    rusage r{};
    if (getrusage(RUSAGE_SELF, &r)) throw std::runtime_error("Cannot read peak RSS");
    return uint64_t(r.ru_maxrss) * 1024ULL;
}
inline uint64_t free_ram() {
    const long pages = sysconf(_SC_AVPHYS_PAGES), size = sysconf(_SC_PAGESIZE);
    if (pages < 0 || size < 0) throw std::runtime_error("Cannot read available RAM");
    return uint64_t(pages) * uint64_t(size);
}
#endif
#endif
