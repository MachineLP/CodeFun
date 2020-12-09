
```
Given two integers dividend and divisor, divide two integers without using multiplication, division and mod operator.

Return the quotient after dividing dividend by divisor.

The integer division should truncate toward zero.

Example 1:

Input: dividend = 10, divisor = 3
Output: 3
Example 2:

Input: dividend = 7, divisor = -3
Output: -2
Note:

Both dividend and divisor will be 32-bit signed integers.
The divisor will never be 0.
Assume we are dealing with an environment which could only store integers within the 32-bit signed integer range: [−231,  231 − 1]. For the purpose of this problem, assume that your function returns 231 − 1 when the division result overflows.
```

```python
# 可以学习一下效率提升的方法哦，下面采用翻倍累加（指数级）、相减的方法。
class Solution(object):
    def divide(self, dividend, divisor):
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        """
        dividend1, divisor1 = abs(dividend), abs(divisor)
        quotient = 0
        while dividend1 >= divisor1:
            temp, i = divisor1, 1
            while dividend1 >= temp:
                dividend1 -= temp
                temp += temp
                quotient += i
                i += i
        if (dividend < 0 and divisor > 0) or (dividend > 0 and divisor < 0):
            quotient = -quotient
        return min(max(quotient, -pow(2,31)),pow(2,31)-1)
        
```
