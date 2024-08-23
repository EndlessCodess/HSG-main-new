#include <chrono>
#include <ctime>
#include <format>
#include <fstream>
#include <iostream>
#include <semaphore>
#include <thread>
#include <vector>

#include "HSG.h"
#include "universal.h"

std::vector<std::vector<float>> train;
// std::vector<std::vector<float>> test;
// std::vector<std::vector<uint64_t>> neighbors;
// std::vector<std::vector<float>> reference_answer;
std::string name;

// auto available_thread = std::counting_semaphore<>(0);
// auto done_semaphore = std::counting_semaphore<>(1);
// uint64_t done_mission = 0;
// uint64_t missions = 0;
// auto done = std::counting_semaphore<>(0);
// uint64_t concurrency = 0;
// auto build_semaphore = std::counting_semaphore<>(1);
// uint64_t building = 0;
// uint64_t builded = 0;
// uint64_t done_search = 0;
// auto search = std::counting_semaphore<>(0);

void f(HSG::Index &index, std::ostream &test_result)
{
    test_result << "vectors in index: " << index.count << std::endl;
    auto n = std::vector<uint64_t>(1000, 0);
    for (auto i = 0; i < index.vectors.size(); ++i)
    {
        if (index.vectors[i].data != nullptr)
        {
            if (index.vectors[i].short_edge_in.size() < 10000)
            {
                ++n[index.vectors[i].short_edge_in.size() / 10];
            }
            else
            {
                ++n.back();
            }
        }
    }
    for (auto i = 0; i < n.size(); ++i)
    {
        if (n[i] != 0)
        {
            test_result << std::format("{0:>6} ~ {1:<6}: {2}", i * 10, (i + 1) * 10, n[i]) << std::endl;
        }

        // test_result << std::format("{0:>6} ~ {1:<6}: {2}", i * 10, (i + 1) * 10, n[i]) << std::endl;
    }
}

void base_test(const uint64_t short_edge_lower_limit, const uint64_t cover_range, const uint64_t build_magnification,
               const uint64_t k)
{
    auto time = std::time(nullptr);
    auto UTC_time = std::gmtime(&time);
    auto test_result = std::ofstream(
        std::format("result/HSG/{0}-{1}-{2}-{3}.txt", name, short_edge_lower_limit, cover_range, build_magnification),
        std::ios::app | std::ios::out);

    test_result << UTC_time->tm_year + 1900 << "年" << UTC_time->tm_mon + 1 << "月" << UTC_time->tm_mday << "日"
                << UTC_time->tm_hour + 8 << "时" << UTC_time->tm_min << "分" << UTC_time->tm_sec << "秒" << std::endl;

    test_result << std::format("short edge lower limit: {0:<4}", short_edge_lower_limit) << std::endl;
    test_result << std::format("cover range: {0:<4}", cover_range) << std::endl;
    test_result << std::format("build magnification: {0:<4}", build_magnification) << std::endl;
    test_result << std::format("top k: {0:<4}", k) << std::endl;

    auto search_magnifications = std::vector<uint64_t>{30, 50, 100, 200, 400};

    test_result << "search magnifications: [" << search_magnifications[0];

    for (auto i = 1; i < search_magnifications.size(); ++i)
    {
        test_result << ", " << search_magnifications[i];
    }

    test_result << "]" << std::endl;

    HSG::Index index(Space::Metric::Euclidean2, train[0].size(), short_edge_lower_limit, cover_range,
                     build_magnification);

    uint64_t build_time = 0;

    for (auto i = 0; i < train.size(); ++i)
    {
        auto begin = std::chrono::high_resolution_clock::now();

        HSG::Add(index, i, train[i].data());

        auto end = std::chrono::high_resolution_clock::now();

        build_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

        if (1000000 <= i && i < 10000000 && (i + 1) % 1000000 == 0)
        {
            auto s = "add " + std::to_string(i + 1 - 1000000) + " to " + std::to_string(i + 1) + " costs: ";

            test_result << std::format("{0:<31}{1:>7} us", s, build_time / 1000000) << std::endl;
            build_time = 0;
        }
        else if (100000 <= i && i < 1000000 && (i + 1) % 100000 == 0)
        {
            auto s = "add " + std::to_string(i + 1 - 100000) + " to " + std::to_string(i + 1) + " costs: ";

            test_result << std::format("{0:<31}{1:>7} us", s, build_time / 100000) << std::endl;
            build_time = 0;
        }
        else if (10000 <= i && i < 100000 && (i + 1) % 10000 == 0)
        {
            auto s = "add " + std::to_string(i + 1 - 10000) + " to " + std::to_string(i + 1) + " costs: ";
            test_result << std::format("{0:<31}{1:>7} us", s, build_time / 10000) << std::endl;
            build_time = 0;
        }
        else if (i == 9999)
        {
            auto s = "add " + std::to_string(0) + " to " + std::to_string(10000) + " costs: ";
            test_result << std::format("{0:<31}{1:>7} us", s, build_time / 10000) << std::endl;
            build_time = 0;
        }
    }

    f(index, test_result);

    // uint64_t se = 0;
    // uint64_t ce = 0;
    // uint64_t le = 0;
    // for (auto i = 0; i < index.vectors.size(); ++i)
    // {
    //     const auto &v = index.vectors[i];
    //     auto t = std::unordered_set<uint64_t>();
    //     for (auto j = v.short_edge_out.begin(); j != v.short_edge_out.end(); ++j)
    //     {
    //         if (i < j->second)
    //         {
    //             t.insert(j->second);
    //         }
    //     }
    //     for (auto j = v.short_edge_in.begin(); j != v.short_edge_in.end(); ++j)
    //     {
    //         if (i < *j)
    //         {
    //             t.insert(*j);
    //         }
    //     }
    //     se += t.size();
    //     t.clear();
    //     for (auto j = v.keep_connected.begin(); j != v.keep_connected.end(); ++j)
    //     {
    //         if (i < *j)
    //         {
    //             ++ce;
    //         }
    //     }
    //     le += v.long_edge_out.size();
    // }
    // test_result << std::format("{0:20}: {1}", "normal short edge", se) << std::endl;
    // test_result << std::format("{0:20}: {1}", "connected short edge", ce) << std::endl;
    // test_result << std::format("{0:20}: {1}", "total short edge", se + ce) << std::endl;
    // test_result << std::format("{0:20}: {1}", "long edge", le) << std::endl;
    // test_result << std::format("{0:20}: {1}", "total edge", se + ce + le) << std::endl;

    // auto cover_rate = HSG::Calculate_Coverage(index);

    // test_result << std::format("cover rate: {0:<6.4}", cover_rate) << std::endl;

    // build_semaphore.acquire();
    // ++builded;
    // if (building == builded)
    // {
    //     search.release();
    // }
    // build_semaphore.release();

    // search.acquire();
    // for (auto i = 0; i < search_magnifications.size(); ++i)
    // {
    //     auto search_magnification = search_magnifications[i];
    //     uint64_t total_hit = 0;
    //     uint64_t total_time = 0;

    //     for (auto i = 0; i < test.size(); ++i)
    //     {
    //         auto begin = std::chrono::high_resolution_clock::now();
    //         auto query_result = HSG::Search(index, test[i].data(), k, search_magnification);
    //         auto end = std::chrono::high_resolution_clock::now();
    //         total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    //         auto hit = verify(train, test[i], reference_answer[i], query_result, k);
    //         total_hit += hit;
    //     }

    //     test_result << std::format("short edge lower limit: {0:<4} ", short_edge_lower_limit);
    //     test_result << std::format("cover range: {0:<4} ", cover_range);
    //     test_result << std::format("build magnification: {0:<4} ", build_magnification);
    //     test_result << std::format("search magnification: {0:<4} ", search_magnification);
    //     test_result << std::format("total hit: {0:<10} ", total_hit);
    //     test_result << std::format("average time: {0:<10}us", total_time / test.size());
    //     test_result << std::endl;
    // }
    // ++done_search;
    test_result.close();
    // done_semaphore.acquire();
    // ++done_mission;
    // if (done_mission == missions)
    // {
    //     done.release();
    // }
    // done_semaphore.release();
    // if (done_search == builded)
    // {
    //     building = 0;
    //     builded = 0;
    //     done_search = 0;
    //     available_thread.release(concurrency);
    // }
    // else
    // {
    //     search.release();
    // }
}

int main(int argc, char **argv)
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(0);
    std::cout.tie(0);

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
        // concurrency = 4;
        bvecs_vectors(argv[1], train, 10000000);
        // bvecs_vectors(argv[2], test);
        // ivecs(argv[3], neighbors);
    }
    else
    {
        if (name == "gist")
        {
            // concurrency = 8;
        }
        else if (name == "fashion-mnist")
        {
            // concurrency = 12;
        }

        train = load_vector(argv[1]);
        // test = load_vector(argv[2]);
        // neighbors = load_neighbors(argv[3]);
    }

    // available_thread.release(concurrency);
    // load_reference_answer(argv[4], reference_answer);

    auto short_edge_lower_limits = std::vector<uint64_t>();
    auto cover_ranges = std::vector<uint64_t>();
    auto build_magnifications = std::vector<uint64_t>();

    auto SELL = std::stringstream(argv[6]);
    auto CR = std::stringstream(argv[7]);
    auto BM = std::stringstream(argv[8]);
    auto k = std::stoull(argv[9]);

    uint64_t temporary = 0;

    while (SELL >> temporary)
    {
        short_edge_lower_limits.push_back(temporary);
    }

    while (CR >> temporary)
    {
        cover_ranges.push_back(temporary);
    }

    while (BM >> temporary)
    {
        build_magnifications.push_back(temporary);
    }

    // missions = short_edge_lower_limits.size() * cover_ranges.size() * build_magnifications.size();

    // for (auto a = 0; a < short_edge_lower_limits.size(); ++a)
    // {
    //     auto &short_edge_lower_limit = short_edge_lower_limits[a];

    //     for (auto b = 0; b < cover_ranges.size(); ++b)
    //     {
    //         auto &cover_range = cover_ranges[b];

    //         for (auto c = 0; c < build_magnifications.size(); ++c)
    //         {
    //             auto &build_magnification = build_magnifications[c];
    //             available_thread.acquire();
    //             build_semaphore.acquire();
    //             ++building;
    //             build_semaphore.release();
    //             auto one_thread = std::thread(base_test, short_edge_lower_limit, cover_range, build_magnification,
    //             k); one_thread.detach();
    //         }
    //     }
    // }

    // done.acquire();
    base_test(8, 4, 50, 10);
    return 0;
}
