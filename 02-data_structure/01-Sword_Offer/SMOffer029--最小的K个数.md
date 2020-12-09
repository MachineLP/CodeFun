```
题目描述

输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。

解题思路

方法一：设一个容器，然后遍历input数组，当容器中的数字的数目不足k个，直接把此时遍历到的数字压入该容器，若容器中数字的数目已经达到k，则比较此时遍历的数字与容器中最大的数字，若小于容器中最大的数字，则将容器中最大的数字替换为此时遍历到的数字。由此我们不难发现大顶堆的数据结构很方便，可以利用STL自带的mutiset（基于红黑树）以及Python中的heapq模块。

```

```C++
class Solution {
public:
    vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
        vector<int>ans;
        multiset<int, greater<int>> inSet;
        if(k < 1 || input.size() < k)
            return ans;
        for(int i = 0; i < k; ++ i)
            inSet.insert(input[i]);
        for(int i = k; i < input.size(); ++ i){
            if(*inSet.begin() > input[i]){
                inSet.erase(inSet.begin());
                inSet.insert(input[i]);
            }
        }
        for(auto it = inSet.begin(); it != inSet.end(); ++ it)
            ans.push_back(*it);
        return ans;
    }
};


```

```python
import heapq
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        # write code here
        ans, h= [], []
        if k < 1 or len(tinput) < k:
            return ans
        for i in range(k):
            h.append(-tinput[i])
        heapq.heapify(h)
        for i in range(k, len(tinput)):
            heapq.heappushpop(h,-tinput[i])
        for i in range(k):
            ans.insert(0, -heapq.heappop(h))
        return ans

import heapq
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        # write code here
        ans = []
        if k < 1 or len(tinput) < k:
            return ans
        ans = heapq.nsmallest(k, tinput)
        return ans


```
