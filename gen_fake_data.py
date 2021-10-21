import numpy as np


def gen_fake_data():
    fake_data = np.random.rand(1024, 32).astype(np.float32)
    # fake_label = np.arange(1).astype(np.int64)
    np.save("fake_data.npy", fake_data)
    # np.save("fake_label.npy", fake_label)


if __name__ == "__main__":
    gen_fake_data()