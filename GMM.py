from sklearn.mixture import GaussianMixture
import numpy as np

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

import pandas as pd

# 原始数据集
X_data = pd.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], columns=['feature1', 'feature2'])
y_data = pd.Series([0, 1, 0, 2, 1])

# 新数据集的大小
sample_num = 1000

# 可以手动设置高斯混合模型的参数，比如高斯分布的数量、协方差矩阵类型等等
n_comp = 3   # 高斯分布的数量

# 将DataFrame和Series转化为NumPy数组
X_data_np = X_data.values
y_data_np = y_data.values

# 使用高斯混合模型拟合原始数据
gmm = GaussianMixture(n_components=n_comp)
gmm.fit(X_data_np, y_data_np)

gmm.weights_ = [0.2,0.3,0.4]

# 生成新的样本数据
np.random.seed(0)   # 设置随机种子
X_data_new, y_data_new = gmm.sample(sample_num)















digits = load_digits()
digits.data.shape

def plot_digits(data):
    fig, ax = plt.subplots(10, 10, figsize=(8, 8),
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8, 8), cmap='binary')
        im.set_clim(0, 16)
plot_digits(digits.data)

from sklearn.decomposition import PCA
pca = PCA(0.99, whiten=True)
data = pca.fit_transform(digits.data)
data.shape

n_components = np.arange(50, 210, 10)
models = [GaussianMixture(n, covariance_type='full', random_state=0)
          for n in n_components]
aics = [model.fit(data).aic(data) for model in models]
plt.plot(n_components, aics);

gmm = GaussianMixture(110, covariance_type='full', random_state=0)
gmm.fit(data)
print(gmm.converged_)

np.random.seed(66)
data_new = gmm.sample(100)
data_new.shape

digits_new = pca.inverse_transform(data_new)
plot_digits(digits_new)

