#pragma once
// Compatibility timer for the otherwise unmodified upstream STREAM source.
#include <windows.h>
struct timeval { long tv_sec; long tv_usec; };
struct timezone { int tz_minuteswest; int tz_dsttime; };
static __inline int gettimeofday(struct timeval *time, struct timezone *zone) {
    LARGE_INTEGER ticks, frequency;
    (void)zone;
    if (!QueryPerformanceFrequency(&frequency) || !QueryPerformanceCounter(&ticks)) return -1;
    time->tv_sec = (long)(ticks.QuadPart / frequency.QuadPart);
    time->tv_usec = (long)((ticks.QuadPart % frequency.QuadPart) * 1000000 / frequency.QuadPart);
    return 0;
}
