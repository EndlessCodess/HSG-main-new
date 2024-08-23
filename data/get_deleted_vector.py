import random
import numpy as np
import struct


def get_irrelevant(neighbors, train_size, k):
    # 将所有的邻居索引放入一个集合中
    neighbors_set = set(neighbors.flatten())
    # 创建一个包含所有训练集索引的集合
    all_indices_set = set(np.arange(train_size))
    # 找出在训练集中但不在邻居集合中的索引
    non_neighbors_indices = all_indices_set - neighbors_set
    print("irrelevant vectors number: {0}".format(len(non_neighbors_indices)))
    if k <= len(non_neighbors_indices):
        print("deleted irrelevant vectors number: {0}".format(k))
        irrelevant = np.random.choice(
            list(non_neighbors_indices), size=k, replace=False
        )
        # 保存为二进制文件
        with open("delete{0}irrelevant.binary".format(k), "wb") as file:
            file.write(struct.pack("Q", k))
            for i in irrelevant:
                file.write(struct.pack("Q", i))
        file.close()


def get_relevant(neighbors, lower_limit, k):
    nl = neighbors.tolist()
    # 记录每行选中的个数
    array = np.full(len(neighbors), 0, dtype="uint64")
    while np.any(array < lower_limit):
        row = np.argmin(array)
        # 从所有元素中随机抽取一个
        selected_element = random.choice(nl[row])
        for row in range(0, len(neighbors)):
            if selected_element in nl[row]:
                array[row] += 1
                nl[row].remove(selected_element)
    # 将所有的邻居索引放入一个集合中
    result = set()
    for i in nl:
        for j in i:
            result.add(j)
    print("{0} relevant vectors can be deleted.".format(len(result)))
    if k < len(result):
        relevant = np.random.choice(list(result), size=k, replace=False)
        # 保存为二进制文件
        with open("delete{0}relevant.binary".format(k), "wb") as file:
            file.write(struct.pack("Q", len(relevant)))
            print("deleted vectors number: {0}".format(len(relevant)))
            for i in relevant:
                file.write(struct.pack("Q", i))
        file.close()
