#include <chrono>
#include <ctime>
#include <format>
#include <iostream>
#include <thread>
#include <vector>

#include "HSG.h"
#include "universal.h"

std::vector<std::vector<float>> train;
std::vector<std::vector<float>> test;
std::vector<std::vector<uint64_t>> neighbors;
std::vector<std::vector<float>> reference_answer;
std::string name;

inline void f1(std::vector<HSG::Offset> &VM, HSG::Offset o)
{
    auto i = std::find(VM.begin(), VM.end(), o);
    auto j = i - VM.begin();
    std::swap(VM[j], VM[VM.size() - 1]);
    VM.pop_back();
}

inline void BFS_Through_SE_1(const HSG::Index &index, const HSG::Offset start_offset, std::vector<HSG::Offset> &VR,
                             std::vector<bool> &VC, std::vector<HSG::Offset> &VM, uint64_t &covered)
{
    auto visited = std::vector<bool>(index.vectors.size(), false);
    visited[start_offset] = true;

    auto last = std::vector<HSG::Offset>();
    last.push_back(start_offset);

    auto next = std::vector<HSG::Offset>();

    for (auto i = 1; i < index.parameters.BFS_round; ++i)
    {
        for (auto j = 0; j < last.size(); ++j)
        {
            const auto &offset = last[j];
            const auto &vector = index.vectors[offset];

            for (auto iterator = vector.short_edge_in.begin(); iterator != vector.short_edge_in.end(); ++iterator)
            {
                const auto &neighbor_offset = *iterator;

                if (!visited[neighbor_offset])
                {
                    visited[neighbor_offset] = true;
                    next.push_back(neighbor_offset);

                    if (!VC[neighbor_offset])
                    {
                        ++covered;
                        VC[neighbor_offset] = true;
                        f1(VM, neighbor_offset);
                    }
                }
            }

            for (auto iterator = vector.short_edge_out.begin(); iterator != vector.short_edge_out.end(); ++iterator)
            {
                const auto neighbor_offset = iterator->second;

                if (!visited[neighbor_offset])
                {
                    visited[neighbor_offset] = true;
                    next.push_back(neighbor_offset);

                    if (!VC[neighbor_offset])
                    {
                        ++covered;
                        VC[neighbor_offset] = true;
                        f1(VM, neighbor_offset);
                    }
                }
            }

            for (auto iterator = vector.keep_connected.begin(); iterator != vector.keep_connected.end(); ++iterator)
            {
                const auto neighbor_offset = *iterator;

                if (!visited[neighbor_offset])
                {
                    visited[neighbor_offset] = true;
                    next.push_back(neighbor_offset);

                    if (!VC[neighbor_offset])
                    {
                        ++covered;
                        VC[neighbor_offset] = true;
                        f1(VM, neighbor_offset);
                    }
                }
            }
        }

        std::swap(last, next);
        next.clear();
    }

    for (auto i = 0; i < last.size(); ++i)
    {
        const auto offset = last[i];
        const auto &vector = index.vectors[offset];

        for (auto iterator = vector.short_edge_in.begin(); iterator != vector.short_edge_in.end(); ++iterator)
        {
            const auto neighbor_offset = *iterator;

            if (!VC[neighbor_offset])
            {
                ++covered;
                VC[neighbor_offset] = true;
                f1(VM, neighbor_offset);
            }
        }

        for (auto iterator = vector.short_edge_out.begin(); iterator != vector.short_edge_out.end(); ++iterator)
        {
            const auto neighbor_offset = iterator->second;

            if (!VC[neighbor_offset])
            {
                ++covered;
                VC[neighbor_offset] = true;
                f1(VM, neighbor_offset);
            }
        }

        for (auto iterator = vector.keep_connected.begin(); iterator != vector.keep_connected.end(); ++iterator)
        {
            const auto neighbor_offset = *iterator;

            if (!VC[neighbor_offset])
            {
                ++covered;
                VC[neighbor_offset] = true;
                f1(VM, neighbor_offset);
            }
        }
    }
}

void O_test(const uint64_t short_edge_limit, const uint64_t cover_range, const uint64_t build_magnification,
            const uint64_t k, const uint64_t search_magnification)
{
    auto time = std::time(nullptr);
    auto UTC_time = std::gmtime(&time);

    auto test_result =
        std::ofstream(std::format("result/HSG/O-{0}-{1}-{2}-{3}-{4}-{5}.txt", name, UTC_time->tm_year + 1900,
                                  UTC_time->tm_mon + 1, UTC_time->tm_mday, UTC_time->tm_hour + 8, UTC_time->tm_min),
                      std::ios::app | std::ios::out);

    test_result << UTC_time->tm_year + 1900 << "年" << UTC_time->tm_mon + 1 << "月" << UTC_time->tm_mday << "日"
                << UTC_time->tm_hour + 8 << "时" << UTC_time->tm_min << "分" << UTC_time->tm_sec << "秒" << std::endl;

    test_result << std::format("short edge limit: {0:<4}", short_edge_limit) << std::endl;
    test_result << std::format("cover range: {0:<4}", cover_range) << std::endl;
    test_result << std::format("build magnification: {0:<4}", build_magnification) << std::endl;
    test_result << std::format("search magnification: {0:<4}", search_magnification) << std::endl;
    test_result << std::format("top k: {0:<4}", k) << std::endl;

    HSG::Index index(Space::Metric::Euclidean2, train[0].size(), short_edge_limit, cover_range, build_magnification);

    index.parameters.BFS_round = std::numeric_limits<uint64_t>::max();

    for (auto i = 0; i < train.size(); ++i)
    {
        HSG::Add(index, i, train[i].data());
    }

    index.parameters.BFS_round = cover_range;

    auto VC = std::vector<bool>(index.vectors.size(), false);
    auto VR = std::vector<uint64_t>();
    auto VM = std::vector<uint64_t>(index.vectors.size() - 1);
    for (auto i = 1; i < index.vectors.size(); ++i)
    {
        VM[i - 1] = i;
    }

    VR.push_back(0);
    VC[0] = true;

    uint64_t covered = 1;
    float rate = 0;

    BFS_Through_SE_1(index, 0, VR, VC, VM, covered);

    while (rate < 0.99)
    {
        uint64_t total_hit = 0;
        uint64_t total_time = 0;

        for (auto i = 0; i < test.size(); ++i)
        {
            auto begin = std::chrono::high_resolution_clock::now();
            auto query_result = HSG::Search(index, test[i].data(), k, search_magnification);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            auto hit = verify(train, test[i], reference_answer[i], query_result, k);
            total_hit += hit;
        }

        rate = (float)covered / index.count;

        std::cout << "The number of vertices that can be reached through long edges: " << VR.size() << std::endl;
        std::cout << "Number of vertices covered: " << covered - VR.size() << std::endl;
        std::cout << "Coverage rate: " << rate << std::endl;
        std::cout << "The number of vertices not covered: " << VM.size() << std::endl;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(0, VM.size() - 1);

        uint64_t offset = dist(gen);
        const auto id = index.vectors[offset].id;
        auto long_path = std::vector<std::pair<float, uint64_t>>();

        std::cout << std::format("Vertices with added long edges(id, offset): ({0}, {1})", id, offset) << std::endl;

        VR.push_back(offset);
        VC[offset] = true;
        f1(VM, offset);
        BFS_Through_SE_1(index, offset, VR, VC, VM, covered);
        Search_Optimize(index, offset, long_path);
        Add_Long_Edges(index, long_path, offset);

        test_result << std::format("cover rate: {0:<.6} total hit: {1:<10} average time: {2:<10}us", rate, total_hit,
                                   total_time / test.size())
                    << std::endl;
    }
}

int main(int argc, char **argv)
{
#if defined(__AVX512F__)
    std::cout << "AVX512 supported. " << std::endl;
#elif defined(__AVX__)
    std::cout << "AVX supported. " << std::endl;
#elif defined(__SSE__)
    std::cout << "SSE supported. " << std::endl;
#else
    std::cout << "no SIMD supported. " << std::endl;
#endif

    std::cout << "CPU physical units: " << std::thread::hardware_concurrency() << std::endl;

    name = std::string(argv[5]);

    if (name == "sift10M")
    {
        bvecs_vectors(argv[1], train, 10000000);
        bvecs_vectors(argv[2], test);
        ivecs(argv[3], neighbors);
    }
    else
    {
        train = load_vector(argv[1]);
        test = load_vector(argv[2]);
        neighbors = load_neighbors(argv[3]);
    }

    load_reference_answer(argv[4], reference_answer);

    auto short_edge_limit = std::stoull(argv[6]);
    auto cover_range = std::stoull(argv[7]);
    auto build_magnification = std::stoull(argv[8]);
    auto search_magnification = std::stoull(argv[9]);
    auto k = std::stoull(argv[10]);

    O_test(short_edge_limit, cover_range, build_magnification, k, search_magnification);

    return 0;
}
