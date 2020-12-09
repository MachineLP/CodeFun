```
题目描述

一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。

解题思路

几乎是上一题斐波拉切数列的翻版，对O(logn)解法感兴趣的也可以去看看上一篇
```

```C++
class Solution {
public:
    int jumpFloor(int number) {
        int ans[2] = {1, 1}, result;
        if(number < 2)
            return ans[number];
        for(int i = 2; i <= number; ++ i){
            result = ans[0] + ans[1];
            ans[0] = ans[1];
            ans[1] = result;
        }
        return result;
    }
};


```

```python
# -*- coding:utf-8 -*-
class Solution:
    def jumpFloor(self, number):
        # write code here
        if number <= 1:
            return number
        first, second, third = 0, 1, 1
        for i in range(number):
            third = first + second
            first = second
            second = third
        return third

```
