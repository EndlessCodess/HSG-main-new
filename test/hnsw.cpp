#include <chrono>
#include <ctime>
#include <format>
#include <fstream>
#include <iostream>
#include <semaphore>
#include <thread>
#include <vector>

#include "../../hnswlib/hnswlib/hnswlib.h"
#include "universal.h"

std::vector<std::vector<float>> train;
std::vector<std::vector<float>> test;
std::vector<std::vector<uint64_t>> neighbors;
std::vector<std::vector<float>> reference_answer;
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

void test_hnsw(uint64_t M, uint64_t ef_construction)
{
    auto test_result = std::ofstream(std::format("result/hnsw/{0}-{1}-{2}.txt", name, M, ef_construction),
                                     std::ios::app | std::ios::out);

    auto time = std::time(nullptr);
    auto UTC_time = std::gmtime(&time);
    test_result << UTC_time->tm_year + 1900 << "年" << UTC_time->tm_mon + 1 << "月" << UTC_time->tm_mday << "日"
                << UTC_time->tm_hour + 8 << "时" << UTC_time->tm_min << "分" << UTC_time->tm_sec << "秒" << std::endl;

    test_result << std::format("M: {0:<4} ef: {1:<4}", M, ef_construction) << std::endl;
    std::vector<uint64_t> efs{10, 20, 40, 80, 120, 200, 400, 600, 800};
    test_result << "ef: [" << efs[0];

    for (auto i = 1; i < efs.size(); ++i)
    {
        test_result << ", " << efs[i];
    }

    test_result << "]" << std::endl;

    int dim = train[0].size();
    int max_elements = train.size();
    // Initing index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> *alg_hnsw =
        new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    // Add data to index
    for (auto i = 0; i < max_elements; ++i)
    {
        alg_hnsw->addPoint(train[i].data(), i);
    }

    auto max_level = alg_hnsw->maxlevel_;
    // auto size = 0;
    // auto size_one = 0;
    // for (auto i = max_level; 0 < i; --i)
    // {
    //     size_one = 0;
    //     for (auto j = 0; j < max_elements; ++j)
    //     {
    //         if (i <= alg_hnsw->element_levels_[j])
    //         {
    //             auto neighbors = (unsigned int *)alg_hnsw->get_linklist(j, i);
    //             auto count = alg_hnsw->getListCount(neighbors);

    //             for (auto k = 0; k < count; ++k)
    //             {
    //                 if (neighbors[k] < j)
    //                 {
    //                     ++size_one;
    //                 }
    //             }
    //         }
    //     }
    //     test_result << std::format("level {0}: {1:<9} edges", i, size_one) << std::endl;
    //     size += size_one;
    // }
    // size_one = 0;
    // for (auto i = 0; i < max_elements; ++i)
    // {
    //     auto neighbors = (unsigned int *)alg_hnsw->get_linklist0(i);
    //     auto count = alg_hnsw->getListCount(neighbors);

    //     for (auto j = 0; j < count; ++j)
    //     {
    //         if (neighbors[j] < i)
    //         {
    //             ++size_one;
    //         }
    //     }
    // }
    // test_result << std::format("level {0}: {1:<9} edges", 0, size_one) << std::endl;
    // size += size_one;
    // test_result << std::format("total edges: {0}", size) << std::endl;

    auto dis = std::vector<std::vector<uint64_t>>(max_level + 1, std::vector<uint64_t>(1000, 0));

    for (auto i = max_level; 0 < i; --i)
    {
        for (auto j = 0; j < max_elements; ++j)
        {
            if (i <= alg_hnsw->element_levels_[j])
            {
                auto neighbors = (unsigned int *)alg_hnsw->get_linklist(j, i);
                auto count = alg_hnsw->getListCount(neighbors);
                if (count < 10000)
                {
                    ++dis[i][count / 10];
                }
                else
                {
                    ++dis[i].back();
                }
            }
        }
    }

    for (auto i = 0; i < max_elements; ++i)
    {
        auto neighbors = (unsigned int *)alg_hnsw->get_linklist0(i);
        auto count = alg_hnsw->getListCount(neighbors);

        if (count < 10000)
        {
            ++dis[0][count / 10];
        }
        else
        {
            ++dis[0].back();
        }
    }

    for (auto i = max_level; 0 <= i; --i)
    {
        test_result << std::format("level {0}:", i) << std::endl;
        for (auto j = 0; j < dis[i].size(); ++j)
        {
            if (dis[i][j] != 0)
            {
                test_result << std::format("    {0:>6} ~ {1:<6}: {2}", j * 10, (j + 1) * 10, dis[i][j]) << std::endl;
            }
        }
    }

    for (auto i = 0; i < dis[0].size(); ++i)
    {
        for (auto j = max_level; 0 < j; --j)
        {
            dis[0][i] += dis[j][i];
        }
    }

    test_result << std::format("total dis:") << std::endl;
    for (auto i = 0; i < dis[0].size(); ++i)
    {
        if (dis[0][i] != 0)
        {
            test_result << std::format("    {0:>6} ~ {1:<6}: {2}", i * 10, (i + 1) * 10, dis[0][i]) << std::endl;
        }
    }

    // build_semaphore.acquire();
    // ++builded;
    // if (building == builded)
    // {
    //     search.release();
    // }
    // build_semaphore.release();

    // search.acquire();
    // for (auto &ef : efs)
    // {
    //     uint64_t total_hit = 0;
    //     uint64_t total_time = 0;
    //     alg_hnsw->setEf(ef);

    //     // Query the elements for themselves and measure recall
    //     for (int i = 0; i < test.size(); ++i)
    //     {
    //         auto begin = std::chrono::high_resolution_clock::now();
    //         std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(test[i].data(),
    //         10); auto end = std::chrono::high_resolution_clock::now(); total_time +=
    //         std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count(); total_hit += verify(train,
    //         test[i], reference_answer[i], result, 10);
    //     }

    //     test_result << std::format("M: {0:<4} ", M);
    //     test_result << std::format("ef_construction: {0:<4} ", ef_construction);
    //     test_result << std::format("ef: {0:<4} ", ef);
    //     test_result << std::format("total hit: {0:<10} ", total_hit);
    //     test_result << std::format("average time: {0:<10}us", total_time / test.size()) << std::endl;
    // }
    // ++done_search;
    delete alg_hnsw;
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
        // concurrency = 5;
        bvecs_vectors(argv[1], train, 10000000);
        // bvecs_vectors(argv[2], test);
        // ivecs(argv[3], neighbors);
    }
    else
    {
        if (name == "gist")
        {
            // concurrency = 9;
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

    if (name == "fashion-mnist")
    {
        test_hnsw(12, 500);
    }
    else if (name == "gist")
    {
        test_hnsw(24, 500);
    }
    else if (name == "sift10M")
    {
        test_hnsw(36, 500);
    }

    // std::vector<uint64_t> Ms{4, 8, 12, 16, 24, 36, 48, 64, 96};
    // std::vector<uint64_t> ef_constructions{500};

    // missions = Ms.size() * ef_constructions.size();

    // for (auto &M : Ms)
    // {
    //     for (auto &ef_construction : ef_constructions)
    //     {
    //         available_thread.acquire();
    //         build_semaphore.acquire();
    //         ++building;
    //         build_semaphore.release();
    //         auto one_thread = std::thread(test_hnsw, M, ef_construction);
    //         one_thread.detach();
    //     }
    // }

    // done.acquire();
    return 0;
}
