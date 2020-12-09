
# coding=utf-8
'''
    数据结构与算法
'''

import time 
import numpy as np


# 二分查找
def binary_search(alist, item):
    """
    二分查找，非递归
    1. 2个参数，待查找alist和查找项item
    2. 声明3个变量，first，last，found(返回值)
    3. while条件，first小于等于last，found是false
    4. mid是first和last的中值(整除)；
    5. 三个if条件，相等alist[mid]=item，小于中值换last，大于中值换first；
    6. 返回found，13行
    :param alist: 待查找alist
    :param item: 待查找项item
    :return: 是否找到
    """
    first = 0
    last = len(alist) - 1
    while first <= last:
        mid = (first + last) // 2
        if alist[mid] == item:
            return True
        else:
            if alist[mid] > item:
                last = mid - 1
            else:
                first = mid + 1
    return False


def binary_search_re(alist, item):
    """
    二分查找，递归
    1. if终止条件，长度为0，返回False；
    2. 中点是长度除2，中值判断；
    3. 小于递归前半部分，大于递归后半部分，返回递归函数；
    4. 11行
    :param alist: 待查找alist
    :param item: 待查找项item
    :return: 是否找到
    """
    if len(alist) == 0:
        return False
    mid = len(alist) // 2
    if alist[mid] == item:
        return True
    else:
        if alist[mid] > item:
            return binary_search_re(alist[:mid], item)
        else:
            return binary_search_re(alist[mid + 1:], item)

# 冒泡排序
def bubble_sort(alist):
    """
    冒泡排序
    1. 两次遍历，第1次遍历长度，倒序逐渐减1，每遍历一次，排序一个；
    2. 第2次，正常遍历，少1个值，因为i和i+1；
    3. 当前位大于下一位，交换当前位和下一位；
    4. 4行
    :param alist: 待排序列表
    :return: None，内部排序
    """
    for p_num in range(len(alist) - 1, 0, -1):
        for i in range(p_num):
            if alist[i] > alist[i + 1]:
                alist[i], alist[i + 1] = alist[i + 1], alist[i]


def bubble_sort_short(alist):
    """
    短冒泡排序，增加exchange，额外终止参数
    1. 初始为True，当为False时终止；
    2. 在第2次循环前，设置为False，交换一次就设置为True，一次未交换则触发终止；
    3. 8行，增加5行的exchange操作
    :param alist:
    :return:
    """
    for p_num in range(len(alist) - 1, 0, -1):
        exchange = False
        for i in range(p_num):
            if alist[i] > alist[i + 1]:
                alist[i], alist[i + 1] = alist[i + 1], alist[i]
                exchange = True
        if not exchange:
            # print('提前终止')
            return

# 插入排序
def insert_sort(alist):
    """
    插入排序，子序列逐渐有序
    1. 遍历列表，存储当前值cur_val，设置游标pos
    2. 游标大于0和游标的值大于当前值，则移位，同时游标减1；
    3. 将当前值赋予游标的终止位置；
    4. 7行
    :param alist: 待排序alist
    :return: None
    """
    for i in range(1, len(alist)):
        cur_val = alist[i]
        pos = i  # 游标
        while pos > 0 and alist[pos - 1] > cur_val:
            alist[pos] = alist[pos - 1]
            pos -= 1
        alist[pos] = cur_val  # 最后游标的位置

# 合并排序
def merge_sort(alist):
    """
    归并排序
    1. 递归，结束条件，只有1个元素，return；
    2. mid中心，左右两部分，递归调用merge_sort；
    3. 遍历左右，添加较小的值；遍历其余部分；
    4. 20行
    :param alist:
    :return:
    """
    if len(alist) < 2:
        return
    mid = len(alist) // 2
    left = alist[:mid]
    right = alist[mid:]
    merge_sort(left)
    merge_sort(right)
    i, j, k = 0, 0, 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            alist[k] = left[i]
            i += 1
        else:
            alist[k] = right[j]
            j += 1
        k += 1
    if i < len(left):
        alist[k:] = left[i:]
    if j < len(right):
        alist[k:] = right[j:]


# 快速排序
def quick_sort(alist, fst, lst):
    """
    快速排序
    1. 确定终止条件，起始大于等于终止；
    2. 起始fst和终止lst的位置，枢轴值pivot是第1个值；
    3. 遍历i和j，i是第2个，j是最后一个；
    4. 循环交换，直到i和j交叉；
    5. 枢轴索引取i和j最小的1个；
    6. 交换枢轴位置的值与起始位置的值；
    7. 递归调用左右两部分；
    8. 16行
    :param alist: 待排序列表
    :param fst: 起始idx
    :param lst: 终止idx
    :return: None
    """
    if fst >= lst:
        return
    pivot = alist[fst]
    i, j = fst + 1, lst
    while i <= j:
        while alist[i] < pivot:
            i += 1
        while alist[j] > pivot:
            j -= 1
        if i < j:
            alist[i], alist[j] = alist[j], alist[i]
            i, j = i + 1, j - 1
    p_idx = min(i, j)  # 枢轴索引
    alist[fst], alist[p_idx] = alist[p_idx], alist[fst]
    quick_sort(alist, fst, p_idx - 1)
    quick_sort(alist, p_idx + 1, lst)


def quick_sort_v2(alist):
    """
    快速排序，需要额外空间
    :param alist: 待排序列表
    :return: 排序列表
    """
    if len(alist) <= 1:
        return alist
    pivot = alist[0]
    small = [i for i in alist if i < pivot]
    mid = [i for i in alist if i == pivot]
    large = [i for i in alist if i > pivot]
    return quick_sort_v2(small) + mid + quick_sort_v2(large)

# 希尔排序
def shell_sort(alist):
    """
    希尔排序
    1. 两部分，第1部分，计算增量gap和起始位置s_pos；
    2. 增量是累除2，s_pos是增量的遍历
    3. 增量插入排序，额外传入起始位置和增量；
    4. range的起始由起始位置+增量；
    5. 循环条件为大于等于增量，差值为增量
    6. 12行，增量部分5行，插入部分7行
    :param alist: 待排序alist
    :return: None
    """
    gap = len(alist) // 2
    while gap > 0:
        for s_pos in range(gap):
            insert_sort_gap(alist, s_pos, gap)
        gap = gap // 2


def insert_sort_gap(alist, s_pos, gap):
    """
    带增量的插入排序
    :param alist: 待排序alist
    :param s_pos: 起始位置
    :param gap: 增量
    :return: None
    """
    for i in range(s_pos + gap, len(alist), gap):
        cur_val = alist[i]
        pos = i
        while pos >= gap and alist[pos - gap] > cur_val:
            alist[pos] = alist[pos - gap]
            pos -= gap
        alist[pos] = cur_val

# 选择排序
def selection_sort(alist):
    """
    选择排序，即选择最大值再交换
    1. 依然是2次遍历，第1次反序，第2次正序，注意起始为1，末尾+1；
    2. max_loc存储最大值，默认第0位；
    3. 当loc的值大于max_loc的值时，max_loc重新赋值；
    4. 交换loc和max_loc
    5. 6行
    :param alist: 待排序alist
    :return: None
    """
    for p_num in range(len(alist) - 1, 0, -1):
        max_loc = 0
        for i in range(1, p_num + 1):
            if alist[i] > alist[max_loc]:
                max_loc = i
        alist[p_num], alist[max_loc] = alist[max_loc], alist[p_num]



def test_of_binary_search():
    test_list = [0, 1, 2, 8, 13, 17, 19, 32, 42]
    print(binary_search(test_list, 3))
    print(binary_search(test_list, 13))
    print(binary_search_re(test_list, 3))
    print(binary_search_re(test_list, 13))


if __name__ == '__main__':
    test_list = np.random.rand(10000000)
    start_time = time.time()
    binary_search(test_list, test_list[9999])
    end_time = time.time()
    print('binary_search:', end_time - start_time)

    start_time = time.time()
    binary_search_re(test_list, test_list[9999])
    end_time = time.time()
    print('binary_search_re:', end_time - start_time)



