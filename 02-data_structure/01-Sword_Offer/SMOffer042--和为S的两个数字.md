```
题目描述

输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。

输出描述:

对应每个测试案例，输出两个数，小的先输出。
解题思路

首先先明确我们的任务，有多对的话，输出乘积最小的一对，如果两个数的和一定，要使得他们的乘积最小，那么两个数应该尽可能相隔的远些，所以我们的任务就变成了找到和为sum的一对数，如果有多对，则数字第一个数字最小的那一对。延续上一题的思路，其实这道题也是双指针法，两个指针i，j分别指向第一个数和最后一个，当这两个数的和大于sum，则前移j，若小于，则后移i。

```

```C++
class Solution {
public:
    vector<int> FindNumbersWithSum(vector<int> array,int sum) {
        vector<int>ans;
        int i = 0, j = array.size() - 1;
        if(array.size() < 2 || (array[0] << 1) > sum || (array[j] << 1) < sum)
            return ans;
        while(i < j){
            int temp = array[i] + array[j];
            if(temp == sum){
                ans.push_back(array[i]);
                ans.push_back(array[j]);
                return ans;
            }
            if(temp > sum)
                -- j;
            else
                ++ i;
        }
        return ans;
    }
};


```

```python
class Solution:
    def FindNumbersWithSum(self, array, tsum):
        # write code here
        if len(array) < 2:
            return []
        i, j = 0, len(array) - 1
        while i < j:
            temp = array[i] + array[j]
            if temp == tsum:
                return[array[i], array[j]]
            elif temp > tsum:
                j -= 1
            else:
                i += 1
        return []


```
