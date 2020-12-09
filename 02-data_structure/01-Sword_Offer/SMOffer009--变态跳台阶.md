```
题目描述

一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。

解题思路

这是一个数学问题，可以用归纳法得出f(n) = 2^(n-1)
```

```C++
class Solution {
public:
    int jumpFloorII(int number) {
        return pow(2, number - 1);
    }
};

```

```python
# -*- coding:utf-8 -*-
class Solution:
    def jumpFloorII(self, number):
        # write code here
        return pow(2, number-1)


```
