```
Given two binary strings, return their sum (also a binary string).

The input strings are both non-empty and contains only characters 1 or 0.

Example 1:

Input: a = "11", b = "1"
Output: "100"
Example 2:

Input: a = "1010", b = "1011"
Output: "10101"
```


```python
class Solution(object):
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        if len(a) > len(b):
            b = '0'* abs(len(a)-len(b)) + b
        else:
            a = '0'* abs(len(b)-len(a)) + a
        
        carry = 0 
        res = ''
        for i in range(len(a)-1,-1,-1):
            temp = int(a[i])+int(b[i])+carry
            
            if temp > 1:
                if temp %2 == 0:
                    res =  '0' + res 
                    carry = 1
                else:
                    res = '1'+ res
                    carry = 1
            else:
                res = str(temp) + res
                carry = 0
    
        if carry == 1:
            res = '1'+ res
        return res
    
```
