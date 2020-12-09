```
题目描述

大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0）。

n<=39

解题思路

斐波拉切数列的递归解法很容易弄懂，但是递归解法的重复运算次数过多，所以我们需要用循环的方式，将前两项保存，求下一项，再更新前两项。
```

```C++
class Solution {
public:
    int Fibonacci(int n) {
        int ans[2] = {0, 1}, result;
        if(n < 2)
            return ans[n];
        for(int i = 2; i <= n; ++ i){
            result = ans[0] + ans[1];
            ans[0] = ans[1];
            ans[1] = result;
        }
        return result;
    }
};


class Solution {
public:
    int Fibonacci(int n) {
        if(n <= 1)
            return n;
        vector<int>ans = matrixMulti(n-1);
        return ans[0];
    }
    vector<int> matrixMulti(int n){
        vector<int>ans(4);
        if(n == 1){
            ans[0] = 1; ans[1] = 1; ans[2] = 1; ans[3] = 0;
            return ans;
        }
        if(n % 2 == 0){
            vector<int>tempans = matrixMulti(n / 2);
            ans[0] = tempans[0] * tempans[0] + tempans[1] * tempans[2];
            ans[1] = tempans[0] * tempans[1] + tempans[1] * tempans[3];
            ans[2] = tempans[0] * tempans[2] + tempans[3] * tempans[2];
            ans[3] = tempans[1] * tempans[2] + tempans[3] * tempans[3];
        }
        else{
            vector<int>tempans = matrixMulti(n / 2);
            ans[0] = tempans[0] * tempans[0] + tempans[1] * tempans[2];
            ans[1] = tempans[0] * tempans[1] + tempans[1] * tempans[3];
            ans[2] = tempans[0] * tempans[2] + tempans[3] * tempans[2];
            ans[3] = tempans[1] * tempans[2] + tempans[3] * tempans[3];
            swap(ans[0], ans[1]);
            swap(ans[2], ans[3]);
            ans[0] += ans[1];
            ans[2] += ans[3];
        }
        return ans;
    }
};


```

```python
# -*- coding:utf-8 -*-
class Solution:
    def Fibonacci(self, n):
        # write code here
        first = 0
        second = 1
        third = 1
        if n <= 1:
            return n
        for i in range(n-1):
            third = first + second
            first = second
            second = third
        return third

```
