import numpy as np
import operator as op
import matplotlib.pyplot as plt
from minst import *


if __name__ == '__main__':
    train_images = load_train_images()
    train_labels = load_train_labels()
    # test_images = load_test_images()
    # test_labels = load_test_labels()

    # 查看前十个数据及其标签以读取是否正确
for i in range(10):
    print(train_labels[i])
    plt.imshow(train_images[i], cmap='gray')
    plt.pause(0.000001)
    plt.show()
print('done')

