```
题目描述

HZ偶尔会拿些专业问题来忽悠那些非计算机专业的同学。今天测试组开完会后,他又发话了:在古老的一维模式识别中,常常需要计算连续子向量的最大和,当向量全为正数的时候,问题很好解决。但是,如果向量中包含负数,是否应该包含某个负数,并期望旁边的正数会弥补它呢？例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。给一个数组，返回它的最大连续子序列的和，你会不会被他忽悠住？(子向量的长度至少是1)

解题思路

这题应该是贪心法最经典的一道题了，遍历数组，用sum记录前面的和，ans记录当前序列最大值，当sum为负的时候，则舍弃前面的项，即令sum为0。

```

```C++
class Solution {
public:
    int FindGreatestSumOfSubArray(vector<int> array) {
        int ans = array[0], sum = array[0];
        for(int i = 1; i < array.size(); ++ i){
            sum += array[i];
            ans = max(sum, ans);
            if(sum < 0)
                sum = 0;
        }
        return ans;
    }
};


```

```python
class Solution:
    def FindGreatestSumOfSubArray(self, array):
        # write code here
        sum, ans = array[0], array[0]
        for i in range(1, len(array)):
            sum += array[i]
            ans = max(sum, ans)
            if sum < 0 :
                sum = 0
        return ans

```
