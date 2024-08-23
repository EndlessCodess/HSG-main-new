#pragma once

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "space.h"

namespace HSG
{

    // 向量内部的唯一标识符
    // 顶点偏移量
    // index.vectors[offset]
    using Offset = uint64_t;

    // 向量外部的唯一标识符
    using ID = uint64_t;

    constexpr uint64_t U64MAX = std::numeric_limits<uint64_t>::max();
    constexpr float F32MAX = std::numeric_limits<float>::max();

    // 向量
    class Vector
    {
      public:
        // 向量的外部id
        ID id;
        // 向量内部的唯一标识符
        Offset offset;
        // 向量的数据
        const float *data;
        //
        float zero;
        // 短的出边
        std::vector<std::pair<float, Offset>> short_edge_out;
        // 短的入边
        std::vector<Offset> short_edge_in;
        // 长的出边
        std::vector<Offset> long_edge_out;
        // 长的入边
        std::pair<Offset, float> long_edge_in;
        //
        std::vector<Offset> keep_connected;

        explicit Vector(const ID id, Offset offset, const float *const data_address, const float zero)
            : id(id), offset(offset), data(data_address), zero(zero), long_edge_in(std::make_pair(U64MAX, 0))
        {
        }
    };

    class Index_Parameters
    {
      public:
        // 向量的维度
        const uint64_t dimension;
        // 距离类型
        const Space::Metric space_metric;
        //
        uint64_t candidate_limit;
        // 短边数量限制
        uint64_t short_edge_limit;
        // 覆盖范围
        uint64_t BFS_round;

        explicit Index_Parameters(const uint64_t dimension, const Space::Metric space_metric,
                                  const uint64_t candidate_limit, const uint64_t short_edge_limit,
                                  const uint64_t BFS_round)
            : dimension(dimension), space_metric(space_metric), candidate_limit(candidate_limit),
              short_edge_limit(short_edge_limit), BFS_round(BFS_round)
        {
            if (candidate_limit < short_edge_limit)
            {
                this->candidate_limit = short_edge_limit;
            }
        }
    };

    // 索引
    //
    // 索引使用零点作为默认起始点
    //
    // 使用64位无符号整形的最大值作为零点的id
    //
    // 所以向量的id应大于等于0且小于64位无符号整形的最大值
    class Index
    {
      public:
        // 索引的参数
        Index_Parameters parameters;
        // 距离计算
        float (*similarity)(const float *vector1, const float *vector2, uint64_t dimension);
        // 索引中向量的数量
        uint64_t count;
        // 索引中的向量
        std::vector<Vector> vectors;
        // 记录存放向量的数组中的空位
        std::stack<Offset> empty;
        // 零点向量
        std::vector<float> zero;
        // 记录向量的 id 和 offset 的对应关系
        std::unordered_map<ID, Offset> id_to_offset;

        explicit Index(const Space::Metric space, const uint64_t dimension, const uint64_t short_edge_limit,
                       const uint64_t BFS_round, const uint64_t candidate_limit)
            : parameters(dimension, space, candidate_limit, short_edge_limit, BFS_round),
              similarity(Space::get_similarity(space)), count(1), zero(dimension, 0.0)
        {
            this->vectors.push_back(Vector(U64MAX, 0, this->zero.data(), 0));
        }
    };

    enum class Operator : uint64_t
    {
        Add,
        Delete,
        Back,
        In,
        Out,
        Default
    };

    inline void Add_Short_Edge(Index &index, const Offset from, const Offset to, const float distance, const Operator O)
    {
        auto &from_vector = index.vectors[from];
        auto &to_vector = index.vectors[to];

        if (O == Operator::Add)
        {
            from_vector.short_edge_out.emplace(from_vector.short_edge_out.begin(), distance, to);
        }
        else
        {
            const auto iterator = std::upper_bound(from_vector.short_edge_out.begin(), from_vector.short_edge_out.end(),
                                                   std::make_pair(distance, 0));

            from_vector.short_edge_out.emplace(iterator, distance, to);
        }

        const auto iterator = std::upper_bound(to_vector.short_edge_in.begin(), to_vector.short_edge_in.end(), from);

        to_vector.short_edge_in.insert(iterator, from);
    }

    inline void Delete_Short_Edge(Index &index, const Offset from, const Offset to, const Operator O)
    {
        auto &from_vector = index.vectors[from];
        auto &to_vector = index.vectors[to];

        if (O == Operator::Back)
        {
            from_vector.short_edge_out.pop_back();

            const auto iterator = std::find(to_vector.short_edge_in.begin(), to_vector.short_edge_in.end(), from);

            to_vector.short_edge_in.erase(iterator);
        }
        else if (O == Operator::In)
        {
            const auto iterator = std::find(to_vector.short_edge_in.begin(), to_vector.short_edge_in.end(), from);

            to_vector.short_edge_in.erase(iterator);
        }
        else if (O == Operator::Out)
        {
            auto iterator = from_vector.short_edge_out.begin();

            while (iterator != from_vector.short_edge_out.end() && iterator->second != to)
            {
                ++iterator;
            }

            from_vector.short_edge_out.erase(iterator);
        }
    }

    inline void Add_Long_Edge(Index &index, const Offset from, const Offset to, const float distance)
    {
        auto &from_vector = index.vectors[from];
        auto &to_vector = index.vectors[to];
        const auto iterator = std::upper_bound(from_vector.long_edge_out.begin(), from_vector.long_edge_out.end(), to);

        from_vector.long_edge_out.insert(iterator, to);
        to_vector.long_edge_in.first = from;
        to_vector.long_edge_in.second = distance;
    }

    inline void Delete_Long_Edge(Index &index, const Offset from, const Offset to, const Operator O = Operator::Default)
    {
        auto &from_vector = index.vectors[from];
        auto &to_vector = index.vectors[to];

        if (O == Operator::Out)
        {
            const auto iterator = std::find(from_vector.long_edge_out.begin(), from_vector.long_edge_out.end(), to);

            from_vector.long_edge_out.erase(iterator);
        }
        else if (O == Operator::Default)
        {
            const auto iterator = std::find(from_vector.long_edge_out.begin(), from_vector.long_edge_out.end(), to);

            from_vector.long_edge_out.erase(iterator);
            to_vector.long_edge_in.first = U64MAX;
        }
    }

    inline void Add_Keep_Connected(Index &index, const Offset from, const Offset to)
    {
        auto &from_vector = index.vectors[from];
        auto &to_vector = index.vectors[to];
        const auto iterator1 =
            std::upper_bound(from_vector.keep_connected.begin(), from_vector.keep_connected.end(), to);
        const auto iterator2 = std::upper_bound(to_vector.keep_connected.begin(), to_vector.keep_connected.end(), from);

        from_vector.keep_connected.insert(iterator1, to);
        to_vector.keep_connected.insert(iterator2, from);
    }

    inline void Delete_Keep_Connected(Index &index, const Offset from, const Offset to, const Operator O)
    {
        auto &from_vector = index.vectors[from];
        auto &to_vector = index.vectors[to];

        if (O == Operator::In)
        {
            const auto iterator = std::find(to_vector.keep_connected.begin(), to_vector.keep_connected.end(), from);

            to_vector.keep_connected.erase(iterator);
        }
    }

    inline Offset Get_Offset(const Index &index, const ID id)
    {
        return index.id_to_offset.find(id)->second;
    }

    inline void Delete_Vector(Index &index, const Offset offset)
    {
        auto &vector = index.vectors[offset];

        vector.data = nullptr;
        vector.short_edge_in.clear();
        vector.short_edge_out.clear();
        vector.keep_connected.clear();
        vector.long_edge_in.first = U64MAX;
        vector.long_edge_out.clear();
    }

    inline bool Adjacent(const Index &index, const Offset offset1, const Offset offset2)
    {
        const auto &v1 = index.vectors[offset1];
        const auto &v2 = index.vectors[offset2];

        if (std::binary_search(v1.keep_connected.begin(), v1.keep_connected.end(), offset2))
        {
            return true;
        }

        if (std::binary_search(v1.short_edge_in.begin(), v1.short_edge_in.end(), offset2))
        {
            return true;
        }

        if (std::binary_search(v2.short_edge_in.begin(), v2.short_edge_in.end(), offset2))
        {
            return true;
        }

        return false;
    }

    inline void Prefetch(const float *const data)
    {
#if defined(__SSE__)
        _mm_prefetch(data, _MM_HINT_T0);
#endif
    }

    inline void Search_Through_LEO(const Index &index, const Offset processing_offset, std::vector<char> &visited,
                                   const float *const target_vector,
                                   std::priority_queue<std::pair<float, Offset>> &nearest_neighbors,
                                   std::priority_queue<std::pair<float, Offset>, std::vector<std::pair<float, Offset>>,
                                                       std::greater<>> &waiting_vectors)
    {
        const auto &processing_vector = index.vectors[processing_offset];

        for (auto iterator = processing_vector.long_edge_out.begin(); iterator != processing_vector.long_edge_out.end();
             ++iterator)
        {
            const auto neighbor_offset = *iterator;
            const auto &neighbor_vector = index.vectors[neighbor_offset];

            Prefetch(neighbor_vector.data);
            visited[neighbor_offset] = true;

            const auto distance = index.similarity(target_vector, neighbor_vector.data, index.parameters.dimension);

            if (nearest_neighbors.size() < index.parameters.candidate_limit)
            {
                nearest_neighbors.push({distance, neighbor_offset});
            }
            else if (distance < nearest_neighbors.top().first)
            {
                nearest_neighbors.pop();
                nearest_neighbors.push({distance, neighbor_offset});
            }

            waiting_vectors.push({distance, neighbor_offset});
        }
    }

    inline void Get_Pool_From_LEO(const Index &index, const Offset processing_offset, std::vector<bool> &visited,
                                  std::vector<Offset> &pool)
    {
        const auto &processing_vector = index.vectors[processing_offset];

        for (auto iterator = processing_vector.long_edge_out.begin(); iterator != processing_vector.long_edge_out.end();
             ++iterator)
        {
            const auto neighbor_offset = *iterator;

            visited[neighbor_offset] = true;
            pool.push_back(neighbor_offset);
        }
    }

    uint64_t total_time(0); // 记录总时间
    uint64_t total_hit(0);
    uint64_t total_time1(0); // 记录总时间
    uint64_t total_hit1(0);
    uint64_t total_time2(0); // 记录总时间
    uint64_t total_hit2(0);
    uint64_t total_time3(0); // 记录总时间
    uint64_t total_hit3(0);
    uint64_t total_time4(0); // 记录总时间
    uint64_t total_hit4(0);
    uint64_t total_time5(0); // 记录总时间
    uint64_t total_hit5(0);
    uint64_t total_time6(0); // 记录总时间
    uint64_t total_hit6(0);
    uint64_t total_time7(0); // 记录总时间
    uint64_t total_hit7(0);
    uint64_t total_time8(0); // 记录总时间
    uint64_t total_hit8(0);
    uint64_t total_time9(0); // 记录总时间
    uint64_t total_hit9(0);
    uint64_t total_time0(0); // 记录总时间
    uint64_t total_hit0(0);

    // 测试函数
    //  时间1
    void prefetch_time(const Vector &neighbor_vector)
    {
        auto begin = std::chrono::high_resolution_clock::now();
        Prefetch(neighbor_vector.data);
        auto end = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        ++total_hit;
    }

    // 时间2
    float distance_time(const Index &index, const float *const target_vector, const Vector &neighbor_vector,
                        const uint64_t &dimension)
    {
        auto begin = std::chrono::high_resolution_clock::now();
        float s = index.similarity(target_vector, neighbor_vector.data, dimension);
        auto end = std::chrono::high_resolution_clock::now();
        total_time1 += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        ++total_hit1;
        return s;
    }

    // 时间3
    void waiting_vectors_pop_time(std::priority_queue<std::pair<float, Offset>, std::vector<std::pair<float, Offset>>,
                                                      std::greater<>> &waiting_vectors)
    {
        auto begin = std::chrono::high_resolution_clock::now();
        waiting_vectors.pop();
        auto end = std::chrono::high_resolution_clock::now();
        total_time2 += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        ++total_hit2;
    }

    bool visited_time(const std::vector<char> &visited, const Offset &neighbor_offset)
    {
        auto begin = std::chrono::high_resolution_clock::now();
        bool s = !visited[neighbor_offset];
        auto end = std::chrono::high_resolution_clock::now();
        total_time3 += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        ++total_hit3;
        return s;
    }

    // 标记遍历过
    void visited_bool_time(std::vector<char> &visited, const Offset &neighbor_offset)
    {
        auto begin = std::chrono::high_resolution_clock::now();
        visited[neighbor_offset] = true;
        auto end = std::chrono::high_resolution_clock::now();
        total_time4 += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        ++total_hit4;
    }

    // 加入计算队列
    void pool_push_time(std::vector<Offset> &pool, const Offset &neighbor_offset)
    {
        auto begin = std::chrono::high_resolution_clock::now();
        pool.push_back(neighbor_offset);
        auto end = std::chrono::high_resolution_clock::now();
        total_time5 += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        ++total_hit5;
    }

    bool nearest_size_compare_time(std::priority_queue<std::pair<float, ID>> &nearest_neighbors,
                                   const uint64_t capacity)
    {
        auto begin = std::chrono::high_resolution_clock::now();
        bool s = nearest_neighbors.size() < capacity;
        auto end = std::chrono::high_resolution_clock::now();
        total_time6 += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        ++total_hit6;
        return s;
    }

    bool nearest_top_compare_time(const float &distance, std::priority_queue<std::pair<float, ID>> &nearest_neighbors)
    {
        auto begin = std::chrono::high_resolution_clock::now();
        bool s = distance < nearest_neighbors.top().first;
        auto end = std::chrono::high_resolution_clock::now();
        total_time7 += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        ++total_hit7;
        return s;
    }

    void nearest_push(std::priority_queue<std::pair<float, ID>> &nearest_neighbors, const float &distance,
                      const ID &neighbor_id)
    {
        auto begin = std::chrono::high_resolution_clock::now();
        nearest_neighbors.push({distance, neighbor_id});
        auto end = std::chrono::high_resolution_clock::now();
        total_time8 += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        ++total_hit8;
    }

    void waiting_push(std::priority_queue<std::pair<float, Offset>, std::vector<std::pair<float, Offset>>,
                                          std::greater<>> &waiting_vectors,
                      const float &distance, const Offset &neighbor_offset)
    {
        auto begin = std::chrono::high_resolution_clock::now();
        waiting_vectors.push({distance, neighbor_offset});
        auto end = std::chrono::high_resolution_clock::now();
        total_time9 += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        ++total_hit9;
    }

    void nearest_pop(std::priority_queue<std::pair<float, ID>> &nearest_neighbors)
    {
        auto begin = std::chrono::high_resolution_clock::now();
        nearest_neighbors.pop();
        auto end = std::chrono::high_resolution_clock::now();
        total_time0 += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        ++total_hit0;
    }

    inline void Get_Pool_From_SE(const Index &index, const Offset processing_offset, std::vector<char> &visited,
                                 std::vector<Offset> &pool)
    {
        const auto &processing_vector = index.vectors[processing_offset];

        for (auto iterator = processing_vector.short_edge_out.begin();
             iterator != processing_vector.short_edge_out.end(); ++iterator)
        {
            const auto neighbor_offset = iterator->second;

            // 测试时间4.1
            /*  if (!visited[neighbor_offset]) */
            if (visited_time(visited, neighbor_offset))
            {
                // 测试时间4.1.1
                /* visited[neighbor_offset] = true; */
                visited_bool_time(visited, neighbor_offset);

                // 测试时间4.1.2
                /* pool.push_back(neighbor_offset); */
                pool_push_time(pool, neighbor_offset);
            }
        }

        for (auto iterator = processing_vector.short_edge_in.begin(); iterator != processing_vector.short_edge_in.end();
             ++iterator)
        {
            const auto neighbor_offset = *iterator;

            // 测试时间4.2
            /* if (!visited[neighbor_offset]) */
            if (visited_time(visited, neighbor_offset))
            {
                // 测试时间4.2.1
                /* visited[neighbor_offset] = true; */
                visited_bool_time(visited, neighbor_offset);

                // 测试时间4.2.2
                /* pool.push_back(neighbor_offset); */
                pool_push_time(pool, neighbor_offset);
            }
        }

        for (auto iterator = processing_vector.keep_connected.begin();
             iterator != processing_vector.keep_connected.end(); ++iterator)
        {
            const auto neighbor_offset = *iterator;

            // 测试时间4.3
            /* if (!visited[neighbor_offset]) */
            if (visited_time(visited, neighbor_offset))
            {
                // 测试时间4.3.1
                /* visited[neighbor_offset] = true; */
                visited_bool_time(visited, neighbor_offset);

                // 测试时间4.3.2
                /* pool.push_back(neighbor_offset); */
                pool_push_time(pool, neighbor_offset);
            }
        }
    }

    inline void Similarity(const Index &index, const float *const target_vector, std::vector<Offset> &pool,
                           std::priority_queue<std::pair<float, Offset>> &nearest_neighbors,
                           std::priority_queue<std::pair<float, Offset>, std::vector<std::pair<float, Offset>>,
                                               std::greater<>> &waiting_vectors)
    {
        if (!pool.empty())
        {
            const auto number = pool.size() - 1;

            Prefetch(index.vectors[pool.front()].data);

            for (auto i = 0; i < number; ++i)
            {
                const auto neighbor_offset = pool[i];
                const auto &neighbor_vector = index.vectors[neighbor_offset];
                const auto next_offset = pool[i + 1];
                const auto &next_vector = index.vectors[next_offset];
                const auto distance = index.similarity(target_vector, neighbor_vector.data, index.parameters.dimension);

                if (nearest_neighbors.size() < index.parameters.candidate_limit)
                {
                    nearest_neighbors.push({distance, neighbor_offset});
                    waiting_vectors.push({distance, neighbor_offset});
                }
                else if (distance < nearest_neighbors.top().first)
                {
                    nearest_neighbors.pop();
                    nearest_neighbors.push({distance, neighbor_offset});
                    waiting_vectors.push({distance, neighbor_offset});
                }

                Prefetch(next_vector.data);
            }

            const auto neighbor_offset = pool.back();
            const auto &neighbor_vector = index.vectors[neighbor_offset];
            const auto distance = index.similarity(target_vector, neighbor_vector.data, index.parameters.dimension);

            if (nearest_neighbors.size() < index.parameters.candidate_limit)
            {
                nearest_neighbors.push({distance, neighbor_offset});
                waiting_vectors.push({distance, neighbor_offset});
            }
            else if (distance < nearest_neighbors.top().first)
            {
                nearest_neighbors.pop();
                nearest_neighbors.push({distance, neighbor_offset});
                waiting_vectors.push({distance, neighbor_offset});
            }

            pool.clear();
        }
    }

    inline void Add_Similarity(const Index &index, const float *const target_vector, std::vector<Offset> &pool,
                               std::priority_queue<std::pair<float, Offset>> &nearest_neighbors,
                               std::priority_queue<std::pair<float, Offset>, std::vector<std::pair<float, Offset>>,
                                                   std::greater<>> &waiting_vectors,
                               std::vector<std::pair<Offset, float>> &all)
    {
        if (!pool.empty())
        {
            const auto number = pool.size() - 1;

            Prefetch(index.vectors[pool.front()].data);

            for (auto i = 0; i < number; ++i)
            {
                const auto neighbor_offset = pool[i];
                const auto &neighbor_vector = index.vectors[neighbor_offset];
                const auto next_offset = pool[i + 1];
                const auto &next_vector = index.vectors[next_offset];
                const auto distance = index.similarity(target_vector, neighbor_vector.data, index.parameters.dimension);

                if (nearest_neighbors.size() < index.parameters.candidate_limit)
                {
                    nearest_neighbors.push({distance, neighbor_offset});
                    waiting_vectors.push({distance, neighbor_offset});
                }
                else if (distance < nearest_neighbors.top().first)
                {
                    nearest_neighbors.pop();
                    nearest_neighbors.push({distance, neighbor_offset});
                    waiting_vectors.push({distance, neighbor_offset});
                }

                if (neighbor_vector.short_edge_out.size() < index.parameters.short_edge_limit ||
                    distance < neighbor_vector.short_edge_out.rbegin()->first)
                {
                    all.push_back({neighbor_offset, distance});
                }

                Prefetch(next_vector.data);
            }

            const auto neighbor_offset = pool.back();
            const auto &neighbor_vector = index.vectors[neighbor_offset];
            const auto distance = index.similarity(target_vector, neighbor_vector.data, index.parameters.dimension);

            if (nearest_neighbors.size() < index.parameters.candidate_limit)
            {
                nearest_neighbors.push({distance, neighbor_offset});
                waiting_vectors.push({distance, neighbor_offset});
            }
            else if (distance < nearest_neighbors.top().first)
            {
                nearest_neighbors.pop();
                nearest_neighbors.push({distance, neighbor_offset});
                waiting_vectors.push({distance, neighbor_offset});
            }

            if (neighbor_vector.short_edge_out.size() < index.parameters.short_edge_limit ||
                distance < neighbor_vector.short_edge_out.rbegin()->first)
            {
                all.push_back({neighbor_offset, distance});
            }

            pool.clear();
        }
    }

    // 查询距离目标向量最近的k个向量
    //
    // k = index.parameters.short_edge_lower_limit
    //
    // 返回最近邻和不属于最近邻但是在路径上的顶点
    inline void Add_Search(const Index &index, const Offset new_offset,
                           std::vector<std::pair<float, Offset>> &long_path, uint64_t &short_path_length,
                           std::priority_queue<std::pair<float, Offset>> &nearest_neighbors,
                           std::vector<std::pair<Offset, float>> &all)
    {
        const auto &new_vector = index.vectors[new_offset];

        // 等待队列
        auto waiting_vectors =
            std::priority_queue<std::pair<float, Offset>, std::vector<std::pair<float, Offset>>, std::greater<>>();

        waiting_vectors.push({new_vector.zero, 0});
        nearest_neighbors.push({new_vector.zero, 0});

        const auto &zero_vector = index.vectors.front();

        if (zero_vector.short_edge_out.size() < index.parameters.short_edge_limit ||
            new_vector.zero < zero_vector.short_edge_out.rbegin()->first)
        {
            all.push_back({0, new_vector.zero});
        }

        // 标记是否被遍历过
        auto visited = std::vector<char>(index.vectors.size(), false);

        visited[0] = true;

        // 计算池子
        auto pool = std::vector<Offset>();

        while (!waiting_vectors.empty())
        {
            const auto old_offset = waiting_vectors.top().second;

            long_path.push_back(waiting_vectors.top());
            Search_Through_LEO(index, old_offset, visited, new_vector.data, nearest_neighbors, waiting_vectors);

            const auto new_offset = waiting_vectors.top().second;

            if (old_offset == new_offset)
            {
                break;
            }
        }

        // 阶段二：
        // 利用短边找到和目标向量最近的向量
        while (!waiting_vectors.empty())
        {
            const auto old_distance = waiting_vectors.top().first;
            const auto old_offset = waiting_vectors.top().second;

            waiting_vectors.pop();
            Get_Pool_From_SE(index, old_offset, visited, pool);
            Add_Similarity(index, new_vector.data, pool, nearest_neighbors, waiting_vectors, all);

            const auto new_distance = waiting_vectors.top().first;

            if (old_distance <= new_distance)
            {
                break;
            }

            ++short_path_length;
        }

        // 阶段三：
        // 查找与目标向量相似度最高（距离最近）的k个向量
        while (!waiting_vectors.empty())
        {
            if (nearest_neighbors.top().first < waiting_vectors.top().first)
            {
                break;
            }

            const auto processing_offset = waiting_vectors.top().second;

            waiting_vectors.pop();
            Get_Pool_From_SE(index, processing_offset, visited, pool);
            Add_Similarity(index, new_vector.data, pool, nearest_neighbors, waiting_vectors, all);
        }

        while (index.parameters.short_edge_limit < nearest_neighbors.size())
        {
            nearest_neighbors.pop();
        }
    }

    inline bool Connected(const Index &index, const Offset start, const Offset offset)
    {
        auto visited = std::vector<bool>(index.vectors.size(), false);

        visited[start] = true;

        auto last = std::vector<Offset>();

        last.push_back(start);

        auto next = std::vector<Offset>();

        for (auto round = 0; round < 4; ++round)
        {
            for (auto iterator = last.begin(); iterator != last.end(); ++iterator)
            {
                const auto &t = index.vectors[*iterator];

                for (auto iterator = t.short_edge_in.begin(); iterator != t.short_edge_in.end(); ++iterator)
                {
                    const auto offset_temporary = *iterator;

                    if (!visited[offset_temporary])
                    {
                        visited[offset_temporary] = true;
                        next.push_back(offset_temporary);
                    }
                }

                for (auto iterator = t.short_edge_out.begin(); iterator != t.short_edge_out.end(); ++iterator)
                {
                    const auto offset_temporary = iterator->second;

                    if (!visited[offset_temporary])
                    {
                        visited[offset_temporary] = true;
                        next.push_back(offset_temporary);
                    }
                }

                for (auto iterator = t.keep_connected.begin(); iterator != t.keep_connected.end(); ++iterator)
                {
                    const auto offset_temporary = *iterator;

                    if (!visited[offset_temporary])
                    {
                        visited[offset_temporary] = true;
                        next.push_back(offset_temporary);
                    }
                }
            }

            if (visited[offset])
            {
                return true;
            }

            std::swap(last, next);
            next.clear();
        }

        for (auto iterator = last.begin(); iterator != last.end(); ++iterator)
        {
            const auto &t = index.vectors[*iterator];

            for (auto iterator = t.short_edge_in.begin(); iterator != t.short_edge_in.end(); ++iterator)
            {
                const auto offset_temporary = *iterator;

                visited[offset_temporary] = true;
            }

            for (auto iterator = t.short_edge_out.begin(); iterator != t.short_edge_out.end(); ++iterator)
            {
                const auto offset_temporary = iterator->second;

                visited[offset_temporary] = true;
            }

            for (auto iterator = t.keep_connected.begin(); iterator != t.keep_connected.end(); ++iterator)
            {
                const auto offset_temporary = *iterator;

                visited[offset_temporary] = true;
            }
        }

        if (visited[offset])
        {
            return true;
        }

        return false;
    }

    // 计算角A的余弦值
    //
    // 余弦值没有除以2
    inline float Cosine_Value(const float a, const float b, const float c)
    {
        return (b * b + c * c - a * a) / (b * c);
    }

    // 添加长边
    inline void Add_Long_Edges(Index &index, const std::vector<std::pair<float, Offset>> &long_path,
                               const Offset offset)
    {
        const auto vector_zero = index.vectors[offset].zero;
        float in_distance = 0;
        Offset in_offset = 0;
        auto i = long_path.size() - 1;
        Offset out_offset = 0;
        float out_distance = 0;
        Offset deleted_offset = 0;

        for (; 0 < i; --i)
        {
            const auto D = long_path[i].first;
            const auto neighbor_offset = long_path[i].second;
            const auto neighbor_vector_zero = index.vectors[neighbor_offset].zero;

            if (vector_zero < neighbor_vector_zero)
            {
                continue;
            }

            if (!Adjacent(index, offset, neighbor_offset))
            {
                const auto CV = Cosine_Value(vector_zero, D, neighbor_vector_zero);

                if (0 < CV)
                {
                    in_offset = neighbor_offset;
                    in_distance = D;
                    break;
                }
            }
        }

        Add_Long_Edge(index, in_offset, offset, in_distance);

        for (auto j = i + 1; j < long_path.size(); ++j)
        {
            const auto neighbor1_offset = long_path[j].second;
            const auto neighbor1_vector_zero = index.vectors[neighbor1_offset].zero;

            if (vector_zero < neighbor1_vector_zero)
            {
                const auto D1 = long_path[j].first;
                const auto CV1 = Cosine_Value(neighbor1_vector_zero, vector_zero, D1);
                const auto &V1 = index.vectors[neighbor1_offset];

                const auto D2 = V1.long_edge_in.second;
                const auto neighbor2_offset = long_path[j - 1].second;
                const auto neighbor2_vector_zero = index.vectors[neighbor2_offset].zero;
                const auto CV2 = Cosine_Value(neighbor1_vector_zero, neighbor2_vector_zero, D2);

                if (CV2 < CV1)
                {
                    out_offset = neighbor1_offset;
                    out_distance = D1;
                    deleted_offset = neighbor2_offset;
                    break;
                }
            }
        }

        if (out_offset != 0)
        {
            Delete_Long_Edge(index, deleted_offset, out_offset, Operator::Default);
            Add_Long_Edge(index, offset, out_offset, out_distance);
        }
    }

    inline void Neighbor_Optimize(Index &index, const Offset new_offset, std::vector<std::pair<Offset, float>> &all)
    {
        for (auto i = 0; i < all.size(); ++i)
        {
            const auto neighbor_offset = all[i].first;
            const auto distance = all[i].second;
            auto &neighbor_vector = index.vectors[neighbor_offset];

            Add_Short_Edge(index, neighbor_offset, new_offset, distance, Operator::Default);

            if (index.parameters.short_edge_limit < neighbor_vector.short_edge_out.size())
            {
                const auto NN_offset = neighbor_vector.short_edge_out.back().second;

                Delete_Short_Edge(index, neighbor_offset, NN_offset, Operator::Back);

                if (!std::binary_search(neighbor_vector.short_edge_in.begin(), neighbor_vector.short_edge_in.end(),
                                        NN_offset))
                {
                    if (!Connected(index, neighbor_offset, NN_offset))
                    {
                        Add_Keep_Connected(index, neighbor_offset, NN_offset);
                    }
                }
            }
        }
    }

    // 添加
    inline void Add(Index &index, const ID id, const float *const added_vector_data)
    {
        Offset new_offset = index.vectors.size();
        const auto zero_distance = Space::Euclidean2::zero(added_vector_data, index.parameters.dimension);

        ++index.count;

        if (index.empty.empty())
        {
            // 在索引中创建一个新向量
            index.vectors.push_back(Vector(id, new_offset, added_vector_data, zero_distance));
        }
        else
        {
            new_offset = index.empty.top();
            index.empty.pop();
            index.vectors[new_offset].id = id;
            index.vectors[new_offset].data = added_vector_data;
            index.vectors[new_offset].zero = zero_distance;
        }

        index.id_to_offset.insert({id, new_offset});

        auto nearest_neighbors = std::priority_queue<std::pair<float, Offset>>();
        auto long_path = std::vector<std::pair<float, Offset>>();
        uint64_t short_path_length = 0;
        auto all = std::vector<std::pair<Offset, float>>();

        // 搜索距离新增向量最近的 index.parameters.short_edge_lower_limit 个向量
        // 同时记录搜索路径
        Add_Search(index, new_offset, long_path, short_path_length, nearest_neighbors, all);

        // 添加短边
        while (!nearest_neighbors.empty())
        {
            const auto distance = nearest_neighbors.top().first;
            const auto neighbor_offset = nearest_neighbors.top().second;

            nearest_neighbors.pop();
            Add_Short_Edge(index, new_offset, neighbor_offset, distance, Operator::Add);
        }

        Neighbor_Optimize(index, new_offset, all);

        if (index.parameters.BFS_round * 2 < short_path_length)
        {
            Add_Long_Edges(index, long_path, new_offset);
        }
    }

    inline void Transfer_LEO(Index &index, const Offset whose_offset, const Offset to_offset)
    {
        auto &whose_vector = index.vectors[whose_offset];

        for (auto i = whose_vector.long_edge_out.begin(); i != whose_vector.long_edge_out.end(); ++i)
        {
            const auto distance =
                index.similarity(index.vectors[*i].data, index.vectors[to_offset].data, index.parameters.dimension);
            Add_Long_Edge(index, to_offset, *i, distance);
        }

        whose_vector.long_edge_out.clear();
    }

    inline void Mark_And_Get(const Index &index, const Offset repaired_offset, std::vector<char> &visited,
                             std::vector<Offset> &pool)
    {
        const auto &repaired_vector = index.vectors[repaired_offset];

        for (auto iterator = repaired_vector.short_edge_out.begin(); iterator != repaired_vector.short_edge_out.end();
             ++iterator)
        {
            visited[iterator->second] = true;
        }

        for (auto iterator = repaired_vector.short_edge_out.begin(); iterator != repaired_vector.short_edge_out.end();
             ++iterator)
        {
            const auto &neighbor_vector = index.vectors[iterator->second];

            for (auto iterator = neighbor_vector.short_edge_in.begin(); iterator != neighbor_vector.short_edge_in.end();
                 ++iterator)
            {
                Get_Pool_From_SE(index, *iterator, visited, pool);
            }

            for (auto iterator = neighbor_vector.short_edge_out.begin();
                 iterator != neighbor_vector.short_edge_out.end(); ++iterator)
            {
                Get_Pool_From_SE(index, iterator->second, visited, pool);
            }

            for (auto iterator = neighbor_vector.keep_connected.begin();
                 iterator != neighbor_vector.keep_connected.end(); ++iterator)
            {
                Get_Pool_From_SE(index, *iterator, visited, pool);
            }
        }
    }

    // 删除索引中的向量
    inline void Erase(Index &index, const ID removed_id)
    {
        const auto removed_offset = Get_Offset(index, removed_id);
        auto &removed_vector = index.vectors[removed_offset];

        index.id_to_offset.erase(removed_id);

        // 删除短边的出边
        for (auto iterator = removed_vector.short_edge_out.begin(); iterator != removed_vector.short_edge_out.end();
             ++iterator)
        {
            Delete_Short_Edge(index, removed_offset, iterator->second, Operator::In);
        }

        // 删除短边的入边
        for (auto iterator = removed_vector.short_edge_in.begin(); iterator != removed_vector.short_edge_in.end();
             ++iterator)
        {
            Delete_Short_Edge(index, *iterator, removed_offset, Operator::Out);
        }

        {
            const auto temporary_offset = removed_vector.short_edge_out.begin()->second;
            auto &temporary_vector = index.vectors[temporary_offset];

            for (auto iterator = removed_vector.keep_connected.begin(); iterator != removed_vector.keep_connected.end();
                 ++iterator)
            {
                const auto neighbor_offset = *iterator;
                auto &neighbor_vector = index.vectors[neighbor_offset];

                Delete_Keep_Connected(index, removed_offset, neighbor_offset, Operator::In);

                if (neighbor_offset != temporary_offset &&
                    !std::binary_search(neighbor_vector.short_edge_in.begin(), neighbor_vector.short_edge_in.end(),
                                        temporary_offset) &&
                    !std::binary_search(temporary_vector.short_edge_in.begin(), temporary_vector.short_edge_in.end(),
                                        neighbor_offset))
                {
                    Add_Keep_Connected(index, neighbor_offset, temporary_offset);
                }
            }
        }

        if (removed_vector.long_edge_in.first != U64MAX)
        {
            Delete_Long_Edge(index, removed_vector.long_edge_in.first, removed_offset, Operator::Out);

            if (!removed_vector.long_edge_out.empty())
            {
                Transfer_LEO(index, removed_offset, removed_vector.long_edge_in.first);
            }
        }

        // 补边
        for (auto iterator = removed_vector.short_edge_in.begin(); iterator != removed_vector.short_edge_in.end();
             ++iterator)
        {
            const auto repaired_offset = *iterator;
            auto &repaired_vector = index.vectors[repaired_offset];
            auto visited = std::vector<char>(index.vectors.size(), false);

            visited[repaired_offset] = true;

            auto nearest_neighbors = std::priority_queue<std::pair<float, Offset>>();
            auto waiting_vectors =
                std::priority_queue<std::pair<float, Offset>, std::vector<std::pair<float, Offset>>, std::greater<>>();
            auto pool = std::vector<Offset>();

            Mark_And_Get(index, repaired_offset, visited, pool);
            Get_Pool_From_SE(index, repaired_offset, visited, pool);
            Get_Pool_From_SE(index, removed_offset, visited, pool);
            Similarity(index, repaired_vector.data, pool, nearest_neighbors, waiting_vectors);

            while (!waiting_vectors.empty())
            {
                if (nearest_neighbors.top().second < waiting_vectors.top().first)
                {
                    break;
                }

                const auto processing_offset = waiting_vectors.top().second;

                waiting_vectors.pop();
                Get_Pool_From_SE(index, processing_offset, visited, pool);
                Similarity(index, repaired_vector.data, pool, nearest_neighbors, waiting_vectors);
            }

            if (!nearest_neighbors.empty())
            {
                while (1 < nearest_neighbors.size())
                {
                    nearest_neighbors.pop();
                }

                const auto TO = nearest_neighbors.top().second;
                auto &TV = index.vectors[TO];

                Add_Short_Edge(index, repaired_offset, TO, nearest_neighbors.top().first, Operator::Default);

                if (std::binary_search(repaired_vector.keep_connected.begin(), repaired_vector.keep_connected.end(),
                                       TO))
                {
                    Delete_Keep_Connected(index, repaired_offset, TO, Operator::Default);
                }

                if (TV.long_edge_in.first == repaired_offset)
                {
                    Delete_Long_Edge(index, repaired_offset, TO, Operator::Default);
                    Transfer_LEO(index, TO, repaired_offset);
                }
                else if (repaired_vector.long_edge_in.first == TO)
                {
                    Delete_Long_Edge(index, TO, repaired_offset, Operator::Default);
                    Transfer_LEO(index, repaired_offset, TO);
                }
            }
        }

        Delete_Vector(index, removed_offset);
        --index.count;
        index.empty.push(removed_offset);
    }

    inline void Search_Similarity(const Index &index, const float *const target_vector, const uint64_t capacity,
                                  std::vector<Offset> &pool,
                                  std::priority_queue<std::pair<float, ID>> &nearest_neighbors,
                                  std::priority_queue<std::pair<float, Offset>, std::vector<std::pair<float, Offset>>,
                                                      std::greater<>> &waiting_vectors)
    {
        if (!pool.empty())
        {
            const auto number = pool.size() - 1;

            /* Prefetch(index.vectors[pool.front()].data); */
            prefetch_time(index.vectors[pool.front()]);

            for (auto i = 0; i < number; ++i)
            {
                const auto neighbor_offset = pool[i];
                const auto &neighbor_vector = index.vectors[neighbor_offset];
                const auto neighbor_id = neighbor_vector.id;
                const auto next_offset = pool[i + 1];
                const auto &next_vector = index.vectors[next_offset];

                /* const auto distance = index.similarity(target_vector, neighbor_vector.data,
                 * index.parameters.dimension); */
                const auto distance = distance_time(index, target_vector, neighbor_vector, index.parameters.dimension);

                // 测试时间5.1
                /* if (nearest_neighbors.size()<capacity) */
                if (nearest_size_compare_time(nearest_neighbors, capacity))
                {
                    // 测试时间5.1.2
                    /* nearest_neighbors.push({distance, neighbor_id}); */
                    nearest_push(nearest_neighbors, distance, neighbor_id);

                    // 测试时间5.1.3
                    /* waiting_vectors.push({distance, neighbor_offset}); */
                    waiting_push(waiting_vectors, distance, neighbor_offset);
                }
                /* else if (distance<nearest_neighbors.top().first) */
                else if (nearest_top_compare_time(distance, nearest_neighbors))
                {
                    // 测试时间5.1.1
                    /* nearest_neighbors.pop(); */
                    nearest_pop(nearest_neighbors);

                    // 测试时间5.1.2
                    /* nearest_neighbors.push({distance, neighbor_id}); */
                    nearest_push(nearest_neighbors, distance, neighbor_id);

                    // 测试时间5.1.3
                    /* waiting_vectors.push({distance, neighbor_offset}); */
                    waiting_push(waiting_vectors, distance, neighbor_offset);
                }

                /* Prefetch(next_vector.data); */
                prefetch_time(next_vector);
            }

            const auto neighbor_offset = pool.back();
            const auto &neighbor_vector = index.vectors[neighbor_offset];
            const auto neighbor_id = neighbor_vector.id;
            /* const auto distance = index.similarity(target_vector, neighbor_vector.data, index.parameters.dimension);
             */
            const auto distance = distance_time(index, target_vector, neighbor_vector, index.parameters.dimension);

            if (nearest_neighbors.size() < capacity)
            {
                /* nearest_neighbors.push({distance, neighbor_id}); */
                nearest_push(nearest_neighbors, distance, neighbor_id);
                /* waiting_vectors.push({distance, neighbor_offset}); */
                waiting_push(waiting_vectors, distance, neighbor_offset);
            }
            else if (distance < nearest_neighbors.top().first)
            {
                /* nearest_neighbors.pop(); */
                nearest_pop(nearest_neighbors);
                /* nearest_neighbors.push({distance, neighbor_id}); */
                nearest_push(nearest_neighbors, distance, neighbor_id);
                /* waiting_vectors.push({distance, neighbor_offset}); */
                waiting_push(waiting_vectors, distance, neighbor_offset);
            }

            pool.clear();
        }
    }

    // 查询距离目标向量最近的top-k个向量
    inline std::priority_queue<std::pair<float, ID>> Search(const Index &index, const float *const target_vector,
                                                            const uint64_t top_k, const uint64_t magnification,
                                                            const size_t i)
    {
        Offset last_offset = 1;
        Offset next_offset = 0;
        float minimum_distance = F32MAX;

        while (last_offset != next_offset)
        {
            last_offset = next_offset;

            const auto &processing_vector = index.vectors[last_offset];

            for (auto iterator = processing_vector.long_edge_out.begin();
                 iterator != processing_vector.long_edge_out.end(); ++iterator)
            {
                const auto neighbor_offset = *iterator;
                const auto &neighbor_vector = index.vectors[neighbor_offset];

                // 测试时间1(总的prefetch)
                /* Prefetch(neighbor_vector.data); */
                prefetch_time(neighbor_vector);

                // 测试时间2(总的距离函数计算)
                /* const auto distance = index.similarity(target_vector, neighbor_vector.data,
                 * index.parameters.dimension); */
                const auto distance = distance_time(index, target_vector, neighbor_vector, index.parameters.dimension);

                if (distance < minimum_distance)
                {
                    minimum_distance = distance;
                    next_offset = neighbor_offset;
                }
            }
        }

        const auto capacity = top_k + magnification;

        // 优先队列
        auto nearest_neighbors = std::priority_queue<std::pair<float, ID>>();

        nearest_neighbors.emplace(minimum_distance, index.vectors[next_offset].id);

        // 标记是否被遍历过
        auto visited = std::vector<char>(index.vectors.size(), false);

        visited[0] = true;
        visited[next_offset] = true;

        // 排队队列
        auto waiting_vectors =
            std::priority_queue<std::pair<float, Offset>, std::vector<std::pair<float, Offset>>, std::greater<>>();

        waiting_vectors.emplace(minimum_distance, next_offset);

        // 计算池子
        auto pool = std::vector<Offset>();
        // pool.reserve(100);

        // 阶段二
        // 查找与目标向量相似度最高（距离最近）的top-k个向量
        while (!waiting_vectors.empty())
        {
            if (nearest_neighbors.top().first < waiting_vectors.top().first)
            {
                break;
            }

            const auto processing_offset = waiting_vectors.top().second;

            // 测试时间3
            /* waiting_vectors.pop(); */
            waiting_vectors_pop_time(waiting_vectors);

            // 测试时间4
            Get_Pool_From_SE(index, processing_offset, visited, pool);

            // 测试时间5
            Search_Similarity(index, target_vector, capacity, pool, nearest_neighbors, waiting_vectors);
        }

        if ((i + 1) % 100 == 0)
        {
            std::cout << "prefetch costs: " << total_time << std::endl;
            std::cout << "distance costs: " << total_time1 << std::endl;
            std::cout << "waiting pop costs: " << total_time2 << std::endl;
            std::cout << "visited costs: " << total_time3 << std::endl;      // 重点分析
            std::cout << "visited bool costs: " << total_time4 << std::endl; // 重点分析
            std::cout << "pool push costs: " << total_time5 << std::endl;    // 重点分析
            std::cout << "size compare costs: " << total_time6 << std::endl;
            std::cout << "top compare costs: " << total_time7 << std::endl;
            std::cout << "nearest push costs: " << total_time8 << std::endl;
            std::cout << "waiting push costs: " << total_time9 << std::endl;
            std::cout << "nearest pop costs: " << total_time0 << std::endl;
        }

        return nearest_neighbors;
    }

    // 查询
    // inline std::priority_queue<std::pair<float, uint64_t>> search(const Index
    // &index, const float *const query_vector,
    //                                                               const
    //                                                               uint64_t
    //                                                               top_k,
    //                                                               const
    //                                                               uint64_t
    //                                                               magnification
    //                                                               = 0)
    // {
    //     // if (index.count == 0)
    //     // {
    //     //     throw std::logic_error("Index is empty. ");
    //     // }
    //     // if (query_vector.size() != index.parameters.dimension)
    //     // {
    //     //     throw std::invalid_argument("The dimension of target vector is
    //     not "
    //     //                                 "equality with vectors in index.
    //     ");
    //     // }
    //     return nearest_neighbors(index, query_vector, top_k, magnification);
    // }

    // Breadth First Search through Short Edges.
    inline void BFS_Through_SE(const Index &index, const Offset start_offset, std::vector<bool> &VC)
    {
        auto visited = std::vector<bool>(index.vectors.size(), false);

        visited[start_offset] = true;

        auto last = std::vector<Offset>();

        last.push_back(start_offset);

        auto next = std::vector<Offset>();

        for (auto i = 1; i < index.parameters.BFS_round; ++i)
        {
            for (auto j = 0; j < last.size(); ++j)
            {
                const auto offset = last[j];
                const auto &vector = index.vectors[offset];

                for (auto iterator = vector.short_edge_in.begin(); iterator != vector.short_edge_in.end(); ++iterator)
                {
                    const auto neighbor_offset = *iterator;

                    if (!visited[neighbor_offset])
                    {
                        visited[neighbor_offset] = true;
                        VC[neighbor_offset] = true;
                        next.push_back(neighbor_offset);
                    }
                }

                for (auto iterator = vector.short_edge_out.begin(); iterator != vector.short_edge_out.end(); ++iterator)
                {
                    const auto neighbor_offset = iterator->second;

                    if (!visited[neighbor_offset])
                    {
                        visited[neighbor_offset] = true;
                        VC[neighbor_offset] = true;
                        next.push_back(neighbor_offset);
                    }
                }

                for (auto iterator = vector.keep_connected.begin(); iterator != vector.keep_connected.end(); ++iterator)
                {
                    const auto neighbor_offset = *iterator;

                    if (!visited[neighbor_offset])
                    {
                        visited[neighbor_offset] = true;
                        VC[neighbor_offset] = true;
                        next.push_back(neighbor_offset);
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

                VC[neighbor_offset] = true;
            }

            for (auto iterator = vector.short_edge_out.begin(); iterator != vector.short_edge_out.end(); ++iterator)
            {
                const auto neighbor_offset = iterator->second;

                VC[neighbor_offset] = true;
            }

            for (auto iterator = vector.keep_connected.begin(); iterator != vector.keep_connected.end(); ++iterator)
            {
                const auto neighbor_offset = *iterator;

                VC[neighbor_offset] = true;
            }
        }
    }

    // 通过长边进行广度优先遍历
    //
    //  Breadth First Search Through Long Edges Out.
    inline void BFS_Through_LEO(const Index &index, std::vector<Offset> &VR, std::vector<bool> &VC)
    {
        auto last = std::vector<Offset>();

        last.push_back(0);

        auto next = std::vector<Offset>();

        while (!last.empty())
        {
            for (auto i = 0; i < last.size(); ++i)
            {
                const auto &offset = last[i];
                const auto &vector = index.vectors[offset];

                for (auto iterator = vector.long_edge_out.begin(); iterator != vector.long_edge_out.end(); ++iterator)
                {
                    const auto &neighbor_offset = *iterator;

                    if (!VC[neighbor_offset])
                    {
                        VR.push_back(neighbor_offset);
                        VC[neighbor_offset] = true;
                        next.push_back(neighbor_offset);
                    }
                }
            }

            std::swap(last, next);
            next.clear();
        }
    }

    // 计算覆盖率
    inline float Calculate_Coverage(const Index &index)
    {
        auto VC = std::vector<bool>(index.vectors.size(), false);
        auto VR = std::vector<Offset>();

        VR.push_back(0);
        VC[0] = true;
        BFS_Through_LEO(index, VR, VC);

        for (auto i = VR.begin(); i != VR.end(); ++i)
        {
            BFS_Through_SE(index, *i, VC);
        }

        uint64_t number = 0;

        for (auto i : VC)
        {
            if (i)
            {
                ++number;
            }
        }

        return float(number - 1) / (index.count - 1);
    }

    inline bool Calculate_Benefits(const Index &index, const std::unordered_set<Offset> &missed,
                                   const Offset start_offset, uint64_t &benefits)
    {

        auto visited = std::vector<bool>(index.vectors.size(), false);

        visited[start_offset] = true;

        auto last = std::vector<Offset>();

        last.push_back(start_offset);

        auto next = std::vector<Offset>();
        uint64_t not_missed = 0;

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

                        if (missed.contains(neighbor_offset))
                        {
                            ++benefits;
                        }
                        else
                        {
                            ++not_missed;
                        }
                    }
                }

                for (auto iterator = vector.short_edge_out.begin(); iterator != vector.short_edge_out.end(); ++iterator)
                {
                    const auto &neighbor_offset = iterator->second;

                    if (!visited[neighbor_offset])
                    {
                        visited[neighbor_offset] = true;
                        next.push_back(neighbor_offset);

                        if (missed.contains(neighbor_offset))
                        {
                            ++benefits;
                        }
                        else
                        {
                            ++not_missed;
                        }
                    }
                }

                for (auto iterator = vector.keep_connected.begin(); iterator != vector.keep_connected.end(); ++iterator)
                {
                    const auto &neighbor_offset = *iterator;

                    if (!visited[neighbor_offset])
                    {
                        visited[neighbor_offset] = true;
                        next.push_back(neighbor_offset);

                        if (missed.contains(neighbor_offset))
                        {
                            ++benefits;
                        }
                        else
                        {
                            ++not_missed;
                        }
                    }
                }
            }

            std::swap(last, next);
            next.clear();
        }

        for (auto i = 0; i < last.size(); ++i)
        {
            const auto &offset = last[i];
            const auto &vector = index.vectors[offset];

            for (auto iterator = vector.short_edge_in.begin(); iterator != vector.short_edge_in.end(); ++iterator)
            {
                const auto &neighbor_offset = *iterator;

                if (!visited[neighbor_offset])
                {
                    visited[neighbor_offset] = true;

                    if (missed.contains(neighbor_offset))
                    {
                        ++benefits;
                    }
                    else
                    {
                        ++not_missed;
                    }
                }
            }

            for (auto iterator = vector.short_edge_out.begin(); iterator != vector.short_edge_out.end(); ++iterator)
            {
                const auto &neighbor_offset = iterator->second;

                if (!visited[neighbor_offset])
                {
                    visited[neighbor_offset] = true;

                    if (missed.contains(neighbor_offset))
                    {
                        ++benefits;
                    }
                    else
                    {
                        ++not_missed;
                    }
                }
            }

            for (auto iterator = vector.keep_connected.begin(); iterator != vector.keep_connected.end(); ++iterator)
            {
                const auto &neighbor_offset = *iterator;

                if (!visited[neighbor_offset])
                {
                    visited[neighbor_offset] = true;

                    if (missed.contains(neighbor_offset))
                    {
                        ++benefits;
                    }
                    else
                    {
                        ++not_missed;
                    }
                }
            }
        }

        if (not_missed * 10 < benefits)
        {
            return true;
        }

        return false;
    }

    // 计算为哪个顶点补长边可以覆盖的顶点最多
    inline void Max_Benefits(const Index &index, const std::unordered_set<Offset> &missed, uint64_t &max_benefits,
                             Offset &max_benefit_offset)
    {
        for (auto iterator = missed.begin(); iterator != missed.end(); ++iterator)
        {
            auto offset = *iterator;
            uint64_t benefits = 1;
            auto end = Calculate_Benefits(index, missed, offset, benefits);

            if (max_benefits < benefits)
            {
                max_benefits = benefits;
                max_benefit_offset = offset;

                if (end)
                {
                    break;
                }
            }
        }
    }

    inline void Optimize_Similarity(const Index &index, const float *const target_vector, std::vector<Offset> &pool,
                                    std::priority_queue<std::pair<float, Offset>, std::vector<std::pair<float, Offset>>,
                                                        std::greater<>> &waiting_vectors)
    {
        if (!pool.empty())
        {
            const auto number = pool.size() - 1;

            Prefetch(index.vectors[pool.front()].data);

            for (auto i = 0; i < number; ++i)
            {
                const auto neighbor_offset = pool[i];
                const auto &neighbor_vector = index.vectors[neighbor_offset];
                const auto next_offset = pool[i + 1];
                const auto &next_vector = index.vectors[next_offset];
                const auto distance = index.similarity(target_vector, neighbor_vector.data, index.parameters.dimension);

                Prefetch(next_vector.data);
                waiting_vectors.push({distance, neighbor_offset});
            }

            auto neighbor_offset = pool.back();
            auto &neighbor_vector = index.vectors[neighbor_offset];
            auto distance = index.similarity(target_vector, neighbor_vector.data, index.parameters.dimension);

            waiting_vectors.push({distance, neighbor_offset});
            pool.clear();
        }
    }

    inline void Search_Optimize(const Index &index, const Offset offset,
                                std::vector<std::pair<float, Offset>> &long_path)
    {
        const auto &vector = index.vectors[offset];
        Offset last_offset = 1;
        Offset next_offset = 0;
        float minimum_distance = F32MAX;

        long_path.push_back({vector.zero, 0});

        while (true)
        {
            last_offset = next_offset;

            const auto &processing_vector = index.vectors[last_offset];

            for (auto iterator = processing_vector.long_edge_out.begin();
                 iterator != processing_vector.long_edge_out.end(); ++iterator)
            {
                const auto &neighbor_offset = *iterator;
                const auto &neighbor_vector = index.vectors[neighbor_offset];

                Prefetch(neighbor_vector.data);

                const auto distance = index.similarity(vector.data, neighbor_vector.data, index.parameters.dimension);

                if (distance < minimum_distance)
                {
                    minimum_distance = distance;
                    next_offset = neighbor_offset;
                }
            }

            if (last_offset == next_offset)
            {
                break;
            }
            else
            {
                long_path.emplace_back(minimum_distance, next_offset);
            }
        }
    }

    // 优化索引结构
    inline void Optimize(Index &index)
    {
        auto VC = std::vector<bool>(index.vectors.size(), false);
        auto VR = std::vector<Offset>();
        auto VM = std::vector<Offset>();

        VR.push_back(0);
        VC[0] = true;
        BFS_Through_LEO(index, VR, VC);

        for (auto i = VR.begin(); i != VR.end(); ++i)
        {
            VC[*i] = true;
            BFS_Through_SE(index, *i, VC);
        }

        uint64_t covered = 0;

        for (auto offset = 0; offset < VC.size(); ++offset)
        {
            if (VC[offset])
            {
                ++covered;
            }
            else if (index.vectors[offset].data != nullptr)
            {
                VM.push_back(offset);
            }
        }

        std::cout << "The number of vertices that can be reached through long edges: " << VR.size() << std::endl;
        std::cout << "Number of vertices covered: " << covered - VR.size() << std::endl;
        std::cout << "Coverage rate: " << (float)covered / index.count << std::endl;
        std::cout << "The number of vertices not covered: " << VM.size() << std::endl;

        VR.clear();
        VC.clear();

        // Max_Benefits(index, missed, benefits, offset);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(0, VM.size() - 1);

        Offset offset = dist(gen);
        const auto id = index.vectors[offset].id;
        auto long_path = std::vector<std::pair<float, Offset>>();

        std::cout << std::format("Vertices with added long edges(id, offset): ({0}, {1})", id, offset) << std::endl;
        // std::cout << "Add a long edge to this vertex to cover an additional "
        //              "number of vertices: "
        //           << benefits << std::endl;

        Search_Optimize(index, offset, long_path);
        Add_Long_Edges(index, long_path, offset);
    }

    inline uint64_t Connected(const Index &index)
    {
        auto visited = std::vector<bool>(index.vectors.size(), false);

        visited[0] = true;

        auto last = std::vector<Offset>();

        last.push_back(0);

        auto next = std::vector<Offset>();

        while (!last.empty())
        {
            for (auto iterator = last.begin(); iterator != last.end(); ++iterator)
            {
                const auto &t = index.vectors[*iterator];

                for (auto iterator = t.short_edge_in.begin(); iterator != t.short_edge_in.end(); ++iterator)
                {
                    const auto t1 = *iterator;

                    if (!visited[t1])
                    {
                        visited[t1] = true;
                        next.push_back(t1);
                    }
                }

                for (auto iterator = t.short_edge_out.begin(); iterator != t.short_edge_out.end(); ++iterator)
                {
                    const auto t1 = iterator->second;

                    if (!visited[t1])
                    {
                        visited[t1] = true;
                        next.push_back(t1);
                    }
                }

                for (auto iterator = t.keep_connected.begin(); iterator != t.keep_connected.end(); ++iterator)
                {
                    const auto t1 = *iterator;

                    if (!visited[t1])
                    {
                        visited[t1] = true;
                        next.push_back(t1);
                    }
                }
            }

            std::swap(last, next);
            next.clear();
        }

        uint64_t number = 0;

        for (auto i = 0; i < visited.size(); ++i)
        {
            if (visited[i])
            {
                ++number;
            }
        }

        return index.count - number;
    }

} // namespace HSG
