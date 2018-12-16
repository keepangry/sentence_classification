"""
参考资料：
https://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process

https://blog.csdn.net/slx_share/article/details/83573339

"""

# import matplotlib.pyplot as plt
# import numpy as np
# from itertools import cycle
#
# color_cycle = cycle('kbryg')
# n_variable = 20
# n_sample = 5
#
# plt.figure(figsize=(500, 300))
# sigma_s = np.eye(n_variable)  # 协方差阵为对角矩阵
#
# xs = np.linspace(0, 1, n_variable)  # 构建y对应的特征
# sigma_s = np.exp(-(np.expand_dims(xs, axis=0) - np.expand_dims(xs, axis=1)) ** 2 / 2)  # 依据特征xs构建协方差阵
#
# for _ in range(n_sample):
#     point = np.random.multivariate_normal(np.zeros(n_variable), sigma_s)
#     plt.plot(np.arange(n_variable), point, color=next(color_cycle))
#     plt.scatter(np.arange(n_variable), point)
#
#
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 正确显示中文
# plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# plt.title('不使用核函数的情况', fontsize=24)
# plt.xticks(np.arange(n_variable))
# plt.xlabel('y', fontsize=20)
# plt.show()

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist, squareform


class Kernel:
    # RBF核函数
    def __init__(self, theta):
        self.theta = theta

    def __call__(self, X, Y=None):
        if Y is None:
            dists = pdist(X / self.theta, metric='sqeuclidean')
            K = np.exp(-0.5 * dists)
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            dists = cdist(X / self.theta, Y / self.theta, metric='sqeuclidean')
            K = np.exp(-0.5 * dists)
        return K


class GPR:
    def __init__(self):
        self.K = None
        self.X = None
        self.y = None

    def log_marginal_likelihood(self, theta):
        # 计算对数边缘似然函数
        K = Kernel(theta)
        sigma = K(self.X)
        log_likelihood = np.log(np.linalg.det(sigma)) + \
                         self.y @ np.linalg.inv(sigma) @ self.y + \
                         sigma.shape[0] * np.log(2 * np.pi)
        return - 0.5 * log_likelihood

    def fit(self, X, y):
        # 训练，获取核函数最优参数
        self.X = X
        self.y = y

        def obj_func(theta):
            return - self.log_marginal_likelihood(theta)

        theta_opt = minimize(obj_func, np.array([1.0]), method='BFGS')
        self.K = Kernel(theta_opt.x[0])

    def predict(self, X_pred):
        # 预测，获取y*的条件概率分布
        K_pred = self.K(X_pred)
        K_train = self.K(self.X)
        K_pred_train = self.K(self.X, X_pred)
        K_inv = np.linalg.inv(K_train)
        mu = K_pred_train.T @ K_inv @ self.y
        mu1 = K_pred_train.T.dot(K_inv).dot(self.y)
        sigma = K_pred - K_pred_train.T @ K_inv @ K_pred_train
        return mu, np.diagonal(sigma)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    gpr = GPR()

    # 生成样本数据
    coefs = [6, -2.5, -2.4, -0.1, 0.2, 0.03]


    def f(x):
        """
        黑盒函数
        :param x:
        :return:
        """
        total = 0
        for exp, coef in enumerate(coefs):
            total += coef * (x ** exp)
        return total


    xs = np.linspace(-5.0, 3.5, 100)
    ys = f(xs)

    X_train = np.array([-4, -1.5, 0, 1.5, 2.5, 2.7])
    y_train = f(X_train)

    X_train = X_train.reshape(-1, 1)
    X_pred = np.linspace(-8, 7, 80).reshape((-1, 1))

    gpr.fit(X_train, y_train)
    y_pred, y_std = gpr.predict(X_pred)

    plt.plot(xs, ys, color='k', linewidth=2, label='True')
    plt.scatter(X_train, y_train, color='b', marker='*', linewidths=3, label='Train_data')
    plt.plot(X_pred, y_pred, color='r', label='Pred')

    plt.fill_between(X_pred.reshape(1, -1)[0], y_pred - y_std, y_pred + y_std, color='darkorange',
                     alpha=0.2)
    plt.legend()
    plt.show()
