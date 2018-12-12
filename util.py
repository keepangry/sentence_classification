import numpy as np
import datetime

def time_string():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def score(y_pred, y_val):
    y_pred = np.array(list(map(lambda x: 0 if x[0] < 0.5 else 1, y_pred)))
    return 1 - sum(list(map(lambda x: abs(x), (y_pred - y_val)))) / len(y_pred)


def gene_grid_search_candidates(grid):
    nums = len(grid)
    max = []
    cur = []
    for i in range(nums):
        max.append(len(grid[i]))
        cur.append(0)
    max = np.array(max)
    cur = np.array(cur)
    mul = 1  # 迭代总次数
    for m in max:
        mul = mul * m

    result = []
    for _ in range(mul):
        one = list(np.zeros(nums))
        for i in range(nums):
            one[i] = grid[i][cur[i]]
        cur[0] += 1  # 加1
        for idx in range(nums - 1):
            if cur[idx] == max[idx]:
                cur[idx] = 0
                cur[idx + 1] += 1
        result.append(one)
    return result


if __name__ == '__main__':
    grid = [
        [1, 2, 3],
        [0.1, 0.2, 0.3],
        [10, 20, 30, 40],
    ]
    result = gene_grid_search_candidates(grid)
    print(result)
    print(len(result))
