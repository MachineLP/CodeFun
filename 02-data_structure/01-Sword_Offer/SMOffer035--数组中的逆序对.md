```
题目描述

在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组,求出这个数组中的逆序对的总数P。并将P对1000000007取模的结果输出。 即输出P%1000000007

输入描述:

题目保证输入的数组中没有的相同的数字

数据范围：

对于%50的数据,size<=10^4

对于%75的数据,size<=10^5

对于%100的数据,size<=2*10^5

示例1

输入

1,2,3,4,5,6,7,0

输出

7

解题思路

类似归并排序的思路，将原数组分解为一个数字为一组的子数组，每两个子数组统计是否有逆序对，先用两个指针分别指向两个子数组的末尾，并比较两个指针指向的数字，如果第一个子数组中的数组大于第二个子数组中的数字，则构成逆序对，并且逆序对的数组等于第二个子数组中剩余数组的个数。统计完成后，将两个子数组合并。

这里比较灵活的地方在于，在递归的时候，将copy数组作为下一次递归的data，而原data在下一次递归中则作为copy数组，直接存放归并排序后的数组。
--------------------- 

```

```C++
class Solution {
public:
    int InversePairs(vector<int> data) {
        if (data.size() == 0)
            return 0;
        vector<int>copy = data;
        int cnt = InversePairsCore(data, copy, 0, data.size() - 1);
        return cnt;
    }
    long long int InversePairsCore(vector<int> &data, vector<int> &copy, int start, int end){
        if (start == end){
            copy[start] = data[start];
            return 0;
        }
        int mid = (end + start) >> 1;
        long long int left = InversePairsCore(copy, data, start, mid);
        long long int right = InversePairsCore(copy, data, mid + 1, end), cnt = 0;
        int i = mid, j = end, index = end;
        while(i >= start && j >= mid + 1){
            if(data[i] > data[j]){
                copy[index--] = data[i--];
                cnt += j - mid;
            }
            else
                copy[index--] = data[j--];
        }
        for(; i >= start; --i)
            copy[index--] = data[i];
        for(; j >= mid + 1; -- j)
            copy[index--] = data[j];
        return (left + right + cnt) % 1000000007;
    }
};


```

```python
# -*- coding:utf-8 -*-
class Solution:
    def InversePairs(self, data):
        # write code here
        if len(data) == 0:
            return 0
        copy = [num for num in data]
        cnt = self.InversePairsCore(data, copy, 0, len(data) - 1)
        return cnt % 1000000007
    def InversePairsCore(self, data, copy, start, end):
        if start == end:
            copy[start] = data[start]
            return 0
        mid = (start + end) / 2
        left = self.InversePairsCore(copy, data, start, mid)
        right = self.InversePairsCore(copy, data, mid + 1, end)
        i, j, index, cnt = start, mid+1, start, 0
        while i <= mid and j <= end:
            if data[i] <= data[j]:
                copy[index] = data[i]
                i += 1
            else:
                copy[index] = data[j]
                cnt += mid-i+1
                j += 1
            index += 1
        while i <= mid:
            copy[index] = data[i]
            i += 1
            index += 1
        while j <= end:
            copy[index] = data[j]
            j += 1
            index += 1
        return left + right + cnt


```
