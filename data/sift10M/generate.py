import numpy as np
import sys

sys.path.append("..")
import get_deleted_vector


def ivecs(path):
    with open(path, "rb") as file:
        k = int.from_bytes(file.read(4), byteorder="little")
        file.seek(0, 2)
        end = file.tell()
        number = np.uint64(end / ((k + 1) * 4))
        a = np.empty([number, k], dtype=int)
        file.seek(0, 0)
        for i in range(0, number):
            a[i] = np.fromfile(path, count=k, offset=4, dtype=np.int32)
    file.close()
    return a


def bvecs(path, n=0):
    with open(path, "rb") as file:
        k = int.from_bytes(file.read(4), byteorder="little")
        file.seek(0, 2)
        end = file.tell()
        number = np.uint64(end / (k + 4))
        if n == 0 or number < n:
            n = number
        a = np.empty([n, k], dtype=np.uint8)
        file.seek(0, 0)
        for i in range(0, n):
            a[i] = np.fromfile(path, count=k, offset=4, dtype=np.uint8)
    file.close()
    return a


train = bvecs("bigann_base.bvecs", 10000000)
neighbors = ivecs("gnd/idx_10M.ivecs")
get_deleted_vector.get_irrelevant(neighbors, 10000000, 2500000)
get_deleted_vector.get_relevant(neighbors, 10, 500)
