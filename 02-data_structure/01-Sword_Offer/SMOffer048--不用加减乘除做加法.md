```
题目描述

写一个函数，求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号。

解题思路

不能采用四则运算，就换成计算机中最常见的位运算

首先各位相加不进位，二进制相加的结果与异或是一致的

其次做进位，只有1+1会产生进位，与运算，之后再将结果左移一位，即为要求的进位

最后把两个步骤的结果相加，重复上述步骤，直到不产生进位。

```


```C++
class Solution {
public:
    int Add(int num1, int num2)
    {
        int ans, c;
        do{
            ans = num1 ^ num2;
            c = (num1 & num2) << 1;
            num1 = ans;
            num2 = c;
        }while(num2 != 0);
        return num1;
    }
};


```


```python
class Solution:
    def Add(self, num1, num2):
        # write code here
        MAX = 0x7fffffff
        mask = 0xffffffff
        while num2 != 0:
            ans = num1 ^ num2
            c = (num2 & num1) << 1
            num1, num2 = ans, c
            num1 = num1 & mask
            num2 = num2 & mask
        return num1 if num1 <= MAX else ~(num1 ^ mask)


```
