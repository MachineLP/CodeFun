```
题目描述

我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？

解题思路

不难归纳出f(n) = f(n-1) + f(n-2)的等式，即该问题为斐波拉切数列的问题变体，只不过初始值有所变化。
```

```C++
class Solution {
public:
    int rectCover(int number) {
        int ans[2] = {0, 1}, result;
        if(number < 2)
            return ans[number];
        ans[0] = 1;
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
    def rectCover(self, number):
        # write code here
        if number == 0:
            return 0
        first, second, third = 0, 1, 1
        for i in range(number):
            third = first + second
            first = second
            second = third
        return third


```
