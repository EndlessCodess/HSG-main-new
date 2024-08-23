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
std::vector<uint64_t> irrelevant;
std::vector<uint64_t> relevant;

std::string_view irrelevant_path;
std::string_view relevant_path;

uint64_t short_edge_lower_limit;
uint64_t cover_range;
uint64_t build_magnification;
uint64_t k;
uint64_t search_magnification;

inline HSG::Index deep_copy(HSG::Index index)
{
    return index;
}

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
    }
}

void f1(HSG::Index &index, std::unordered_set<HSG::Offset> &not_delete)
{
    auto next = std::queue<HSG::Offset>();
    next.push(0);
    while (!next.empty())
    {
        auto t = next.front();
        next.pop();
        not_delete.insert(index.vectors[t].id);
        for (auto i = 0; i < index.vectors[t].long_edge_out.size(); ++i)
        {
            next.push(index.vectors[t].long_edge_out[i]);
        }
    }
}

void delete_both()
{
    auto time = std::time(nullptr);
    auto UTC_time = std::gmtime(&time);

    auto test_result =
        std::ofstream(std::format("result/HSG/DB-{0}-{1}-{2}-{3}-{4}-{5}.txt", name, UTC_time->tm_year + 1900,
                                  UTC_time->tm_mon + 1, UTC_time->tm_mday, UTC_time->tm_hour + 8, UTC_time->tm_min),
                      std::ios::app | std::ios::out);

    test_result << UTC_time->tm_year + 1900 << "年" << UTC_time->tm_mon + 1 << "月" << UTC_time->tm_mday << "日"
                << UTC_time->tm_hour + 8 << "时" << UTC_time->tm_min << "分" << UTC_time->tm_sec << "秒" << std::endl;

    test_result << std::format("short edge lower limit: {0:<4}", short_edge_lower_limit) << std::endl;
    test_result << std::format("cover range: {0:<4}", cover_range) << std::endl;
    test_result << std::format("build magnification: {0:<4}", build_magnification) << std::endl;
    test_result << std::format("top k: {0:<4}", k) << std::endl;
    test_result << std::format("search magnification: {0:<4}", search_magnification) << std::endl;
    test_result << std::format("irrelevant number: {0:<9}", irrelevant.size()) << std::endl;
    test_result << std::format("relevant number: {0:<9}", relevant.size()) << std::endl;

    HSG::Index index(Space::Metric::Euclidean2, train[0].size(), short_edge_lower_limit, cover_range,
                     build_magnification);

    for (auto i = 0; i < train.size(); ++i)
    {
        HSG::Add(index, i, train[i].data());
    }

    f(index, test_result);
    auto not_delete = std::unordered_set<HSG::Offset>();
    f1(index, not_delete);

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

        // auto cover_rate = HSG::Calculate_Coverage(index);
        // test_result << std::format("cover rate: {0:<6.4} total hit: {1:<10} average time: {2:<10}us", cover_rate,
        //                            total_hit, total_time / test.size())
        //             << std::endl;
        test_result << std::format("total hit: {0:<10} average time: {1:<10}us", total_hit, total_time / test.size())
                    << std::endl;
    }

    auto deleted = std::unordered_set<uint64_t>();
    uint64_t S = 100000;

    while (true)
    {
        auto t = deep_copy(index);
        uint64_t irrelevant_number = 0;
        uint64_t relevant_number = 0;

        for (auto i = 1; i <= 5; ++i)
        {

            while (irrelevant_number < S * i)
            {
                if (not_delete.contains(irrelevant[irrelevant_number]))
                {++irrelevant_number;
                    continue;
                }
                HSG::Erase(t, irrelevant[irrelevant_number]);
                deleted.insert(irrelevant[irrelevant_number]);
                ++irrelevant_number;
            }

            while (relevant_number < (relevant.size() / 5) * i)
            {
                if (not_delete.contains(relevant[relevant_number]))
                {++relevant_number;
                    continue;
                }
                HSG::Erase(t, relevant[relevant_number]);
                deleted.insert(relevant[relevant_number]);
                ++relevant_number;
            }

            f(index, test_result);

            uint64_t total_hit = 0;
            uint64_t total_time = 0;

            for (auto i = 0; i < test.size(); ++i)
            {
                auto begin = std::chrono::high_resolution_clock::now();
                auto query_result = HSG::Search(t, test[i].data(), k, search_magnification);
                auto end = std::chrono::high_resolution_clock::now();
                total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
                auto hit =
                    verify_with_delete(train, test[i], neighbors[i], reference_answer[i], query_result, deleted, k);
                total_hit += hit;
            }

            // auto cover_rate = HSG::Calculate_Coverage(t);
            // test_result << std::format("cover rate: {0:<6.4} total hit: {1:<10} average time: {2:<10}us", cover_rate,
            //                            total_hit, total_time / test.size())
            //             << std::endl;
            test_result << std::format("total hit: {0:<10} average time: {1:<10}us", total_hit,
                                       total_time / test.size())
                        << std::endl;
        }

        irrelevant_number = 0;
        relevant_number = 0;

        for (auto i = 1; i <= 5; ++i)
        {
            while (irrelevant_number < S * i)
            {
                if (not_delete.contains(irrelevant[irrelevant_number]))
                {++irrelevant_number;
                    continue;
                }
                HSG::Add(t, irrelevant[irrelevant_number], train[irrelevant[irrelevant_number]].data());
                deleted.erase(irrelevant[irrelevant_number]);
                ++irrelevant_number;
            }

            while (relevant_number < (relevant.size() / 5) * i)
            {
                if (not_delete.contains(relevant[relevant_number]))
                {++relevant_number;
                    continue;
                }
                HSG::Add(t, relevant[relevant_number], train[relevant[relevant_number]].data());
                deleted.erase(relevant[relevant_number]);
                ++relevant_number;
            }

            f(index, test_result);

            uint64_t total_hit = 0;
            uint64_t total_time = 0;

            for (auto i = 0; i < test.size(); ++i)
            {
                auto begin = std::chrono::high_resolution_clock::now();
                auto query_result = HSG::Search(t, test[i].data(), k, search_magnification);
                auto end = std::chrono::high_resolution_clock::now();
                total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
                auto hit =
                    verify_with_delete(train, test[i], neighbors[i], reference_answer[i], query_result, deleted, k);
                total_hit += hit;
            }

            // auto cover_rate = HSG::Calculate_Coverage(t);
            // test_result << std::format("cover rate: {0:<6.4} total hit: {1:<10} average time: {2:<10}us", cover_rate,
            //                            total_hit, total_time / test.size())
            //             << std::endl;
            test_result << std::format("total hit: {0:<10} average time: {1:<10}us", total_hit,
                                       total_time / test.size())
                        << std::endl;
        }

        uint64_t again = 0;
        std::cout << "again?" << std::endl;
        std::cin >> again;
        if (again == 0)
        {
            break;
        }
        else
        {
            std::cout << "Ready to load new data?" << std::endl;
            std::cin >> again;

            irrelevant.clear();
            load_deleted(irrelevant_path.data(), irrelevant);
            relevant.clear();
            load_deleted(relevant_path.data(), relevant);
        }
    }

    test_result.close();
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
    load_deleted(argv[11], irrelevant);
    load_deleted(argv[12], relevant);

    std::cout << "delete irrelevant number: " << irrelevant.size() << std::endl;
    std::cout << "delete relevant number: " << relevant.size() << std::endl;

    short_edge_lower_limit = std::stoull(argv[6]);
    cover_range = std::stoull(argv[7]);
    build_magnification = std::stoull(argv[8]);
    search_magnification = std::stoull(argv[9]);
    k = std::stoull(argv[10]);
    irrelevant_path = std::string_view(argv[11]);
    relevant_path = std::string_view(argv[12]);

    delete_both();

    return 0;
}
