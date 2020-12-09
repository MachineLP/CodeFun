
```
Given a 32-bit signed integer, reverse digits of an integer.

Example 1:

Input: 123
Output: 321
Example 2:

Input: -123
Output: -321
Example 3:

Input: 120
Output: 21
Note:
Assume we are dealing with an environment which could only store integers within the 32-bit signed integer range: [−231,  231 − 1]. For the purpose of this problem, assume that your function returns 0 when the reversed integer overflows.
```


```python
class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        overflow_pos=pow(2,31)-1
        overflow_neg=float(-pow(2,31))

        rev=0
        while x!=0:
            trunc=int(float(x)/10)
            pop=x-trunc*10
            x=trunc
            if rev > overflow_pos/10 or (rev==overflow_pos//10 and pop > 7): return 0
            if rev < overflow_neg/10 or (rev==int(overflow_neg/10) and pop < -8): return 0
            rev=rev*10+pop
        return rev
```

