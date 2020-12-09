```
题目描述

给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。

解题思路

本题的高效解法，其实在第5题斐波拉契数列那一题，矩阵乘法的解法中就已经有所体现，即利用下面的公式简化运算。

同时要注意，当n为负数，我们先求a的-n次方，然后取倒数，但是如果此时a为0，则取倒数的时候，会运算错误，所以要另外判断。

```

```C++
class Solution {
public:
    bool flag = false;
    double Power(double base, int exponent) {
        flag = false;
        if(base == 0 && exponent < 0){
            flag = true;
            return 0.0;
        }
        unsigned int absExponent = abs(exponent);
        double ans = PowerWithUnsignedExponent(base, absExponent);
        if(exponent < 0)
            ans = 1.0 / ans;
        return ans;
    }
    double PowerWithUnsignedExponent(double base, unsigned int exponent){
        if(exponent == 0)
            return 1;
        if(exponent == 1)
            return base;
        double result = PowerWithUnsignedExponent(base, exponent >> 1);
        result *= result;
        if(exponent &0x1 == 1)
            result *= base;
        return result;
    }
};


```

```python
class Solution:
    def Power(self, base, exponent):
        # write code here
        if base == 0:
            return 0.0
        absExponent = abs(exponent)
        res = self.PowerWithUnsigenedExponent(base, absExponent)
        if exponent < 0:
            res = 1.0 / res
        return res
 
    def PowerWithUnsigenedExponent(self, base, exponent):
        if exponent == 0:
            return 1
        if exponent == 1:
            return base
        res = self.PowerWithUnsigenedExponent(base, exponent >> 1)
        res *= res
        if exponent & 1 == 1:
            res *= base
        return res


```
