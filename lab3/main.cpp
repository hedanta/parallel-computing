#include <iostream>
#include <fstream>
#include <intrin.h>
#include <iomanip>
#include <sstream>
#include <string>

void print(std::ostream& out, const std::string& text) {
    std::cout << text;
    out << text;
}

void print_feature(std::ostream& out, const std::string& name, bool value) {
    std::stringstream ss;

    ss << " " << std::left << std::setw(25) << name
       << (value ? " supported\n" : " not supported\n");

    std::cout << ss.str();
    out << ss.str();
}

void print_value(std::ostream& out, const std::string& name, const int& value) {
    std::stringstream ss;
    ss << " " << std::left << std::setw(25) << name << " " << value << '\n';
    std::cout << ss.str();
    out << ss.str();
}

int main() {
    std::ofstream fout("cpuid_output.txt");

    int cpuInfo[4];

    // EAX = 0 - vendor
    __cpuid(cpuInfo, 0);

    char vendor[13];
    memcpy(vendor + 0, &cpuInfo[1], 4);
    memcpy(vendor + 4, &cpuInfo[3], 4);
    memcpy(vendor + 8, &cpuInfo[2], 4);
    vendor[12] = '\0';

    print(fout, "Vendor: " + std::string(vendor) + "\n");

    char brand[49] = { 0 };
    for (int i = 0; i < 3; i++) {
        __cpuid(cpuInfo, 0x80000002 + i);
        memcpy(brand + i * 16, cpuInfo, 16);
    }
    print(fout, "Brand: " + std::string(brand) + "\n\n");

    // EAX = 1 - версия cpu
    __cpuid(cpuInfo, 1);

    int eax = cpuInfo[0];
    int ebx = cpuInfo[1];
    int ecx = cpuInfo[2];
    int edx = cpuInfo[3];

    int stepping = eax & 0xF;
    int model = (eax >> 4) & 0xF;
    int family = (eax >> 8) & 0xF;
    int processor_type = (eax >> 12) & 0x3;
    int ext_model = (eax >> 16) & 0xF;
    int ext_family = (eax >> 20) & 0xFF;

    print(fout, "--------=== CPU INFO ===--------\n");
    print_value(fout, "Stepping: ", stepping);
    print_value(fout, "Model: ", model);
    print_value(fout, "Family: ", family);
    print_value(fout, "Processor type: ", processor_type);
    print_value(fout, "Extended model: ", ext_model);
    print_value(fout, "Extended family: ", ext_family);

    int logical_procs = (ebx >> 16) & 0xFF;
    int apic_id = (ebx >> 24) & 0xFF;

    print_value(fout, "Logical processors: ", logical_procs);
    print_value(fout, "APIC ID: ", apic_id);


    print(fout, "\n--------=== BASIC FEATURES ===--------\n");

    print_feature(fout, "SSE",  edx & (1 << 25));
    print_feature(fout, "SSE2", edx & (1 << 26));
    print_feature(fout, "SSE3", ecx & (1 << 0));
    print_feature(fout, "SSSE3", ecx & (1 << 9));
    print_feature(fout, "SSE4.1", ecx & (1 << 19));
    print_feature(fout, "SSE4.2", ecx & (1 << 20));
    print_feature(fout, "AVX", ecx & (1 << 28));
    print_feature(fout, "FMA", ecx & (1 << 12));

    // EAX = 7 - расширения
    __cpuidex(cpuInfo, 7, 0);

    ebx = cpuInfo[1];
    ecx = cpuInfo[2];
    edx = cpuInfo[3];

    print(fout, "\n--------=== EXTENDED FEATURES ===--------\n");

    print_feature(fout, "AVX2", ebx & (1 << 5));
    print_feature(fout, "RTM/TSX", ebx & (1 << 11));
    print_feature(fout, "SHA",  ebx & (1 << 29));
    print_feature(fout, "GFNI", ecx & (1 << 8));
    print_feature(fout, "AVX512F", ebx & (1 << 16));
    print_feature(fout, "AMX-BF16", edx & (1 << 22));
    print_feature(fout, "AMX-TILE", edx & (1 << 24));
    print_feature(fout, "AMX-INT8", edx & (1 << 25));

    __cpuid(cpuInfo, 0x80000000);

    unsigned int max_ext = cpuInfo[0];

    char buffer[100];
    sprintf(buffer, "\n Max extended function: 0x%08X\n", max_ext);
    print(fout, buffer);


    if (max_ext >= 0x80000001) {
        __cpuid(cpuInfo, 0x80000001);

        int ecx = cpuInfo[2];
        int edx = cpuInfo[3];

        print(fout, "\n--------=== AMD extended features ===--------\n");

        print_feature(fout, "SSE4a", ecx & (1 << 6));
        print_feature(fout, "FMA4", ecx & (1 << 16));
        print_feature(fout, "3DNow!", edx & (1 << 31));
        print_feature(fout, "Ext 3DNow!", edx & (1 << 30));
    }
    else {
        print(fout, "[WARNING]: not working on " + std::string(brand) + "\n");
        print(fout, "CPUID 80000001h not supported\n");
    }

    // EAX = 16 - частота
    print(fout, "\n--------=== FREQUENCY (MHz) ===--------\n");

    __cpuid(cpuInfo, 0);
    int max_basic = cpuInfo[0];

    if (max_basic >= 0x16) {
        __cpuid(cpuInfo, 0x16);

        int base_freq = cpuInfo[0] & 0xFFFF;
        int max_freq = cpuInfo[1] & 0xFFFF;
        int bus_freq = cpuInfo[2] & 0xFFFF;

        if (base_freq == 0) {
            print(fout, "[WARNING]: not working on " + std::string(brand) + "\n");
        }
        else {
            print_value(fout, "Base frequency: ", base_freq);
            print_value(fout, "Max frequency: ", max_freq);
            print_value(fout, "Bus frequency: ", bus_freq);
        }
    } 
    else {
        print(fout, "[WARNING]: not working on " + std::string(brand) + "\n");
        print(fout, "CPUID 0x16 not supported\n");
    }

    print(fout, "\n--------=== CACHE INFO ===--------\n");

    int cache_leaf = std::string(vendor) == "GenuineIntel" ? 4 : 0x8000001D;

    for (int i = 0; ; i++) {
        __cpuidex(cpuInfo, cache_leaf, i);

        int eax = cpuInfo[0];
        int ebx = cpuInfo[1];
        int ecx = cpuInfo[2];
        int edx = cpuInfo[3];

        int cache_type = eax & 0x1F;
        if (cache_type == 0) {
            break;
        }

        int cache_level = (eax >> 5) & 0x7;
        int threads = ((eax >> 14) & 0xFFF) + 1;

        int line_size = (ebx & 0xFFF) + 1;
        int partitions = ((ebx >> 12) & 0x3FF) + 1;
        int ways = ((ebx >> 22) & 0x3FF) + 1;

        int sets = ecx + 1;

        int cache_size = (ways * partitions * line_size * sets) / 1024;

        std::string type_str;
        if (cache_type == 1) type_str = " Data cache";
        else if (cache_type == 2) type_str = " Instruction cache";
        else if (cache_type == 3) type_str = " Unified cache";

        print(fout, "\nECX=" + std::to_string(i) + ": ");

        char buffer[100];
        sprintf(buffer, "%08X:%08X:%08X:%08X\n", eax, ebx, ecx, edx);
        print(fout, buffer);

        print(fout, type_str + "\n");
        print_value(fout, "Cache level: ", cache_level);
        print_value(fout, "Threads per cache: ", threads);
        print_value(fout, "Cache line size: ", line_size);
        print_value(fout, "Partitions: ", partitions);
        print_value(fout, "Ways (associativity): ", ways);

        if (edx & (1 << 1))
            print(fout, " Inclusive\n");
        else
            print(fout, " Exclusive\n");

        print_value(fout, "Number of sets: ", sets);
        print_value(fout, "Cache size (KB): ", cache_size);
    }
    fout.close();

    std::cout << "\nSaved to cpuid_output.txt\n";
}