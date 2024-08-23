from typing import List, Tuple, Union
import h5py
import numpy as np
import sys

sys.path.append("..")
import get_deleted_vector


def dataset_transform(
    dataset: h5py.Dataset,
) -> Tuple[
    Union[np.ndarray, List[np.ndarray]],
    Union[np.ndarray, List[np.ndarray]],
    Union[np.ndarray, List[np.ndarray]],
]:
    return (
        np.array(dataset["train"]),
        np.array(dataset["test"]),
        np.array(dataset["neighbors"]),
    )


train, _, neighbors = dataset_transform(h5py.File("gist-960-euclidean.hdf5", "r"))
get_deleted_vector.get_irrelevant(neighbors, len(train), 250000)
get_deleted_vector.get_relevant(neighbors, 10, 25000)
