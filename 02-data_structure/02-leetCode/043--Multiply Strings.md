
```
Given two non-negative integers num1 and num2 represented as strings, return the product of num1 and num2, also represented as a string.

Example 1:

Input: num1 = "2", num2 = "3"
Output: "6"
Example 2:

Input: num1 = "123", num2 = "456"
Output: "56088"
Note:

The length of both num1 and num2 is < 110.
Both num1 and num2 contain only digits 0-9.
Both num1 and num2 do not contain any leading zero, except the number 0 itself.
You must not use any built-in BigInteger library or convert the inputs to integer directly.
```

```python
class Solution(object):
    def __init__(self):
        self.dict_num={'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}
    def multiply(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        if len(num1)<=0 or len(num2)<=0:
            return ''
        result = ''
        for i in range(len(num1)-1,-1,-1):
            carry = 0
            val = ''
            for j in range(len(num2)-1,-1,-1):
                v = ( self.dict_num[num2[j]] ) * ( self.dict_num[num1[i]] ) + carry
                val = str(v%10) + val
                carry = v/10
            if carry: val = str(carry) + val
            for j in range(i,len(num1)-1):
                val = val + '0'
            result = self.str_plus(result, val)
        if result[0]=='0': return '0'
        return result
    
    def str_plus(self, num1, num2):
        if len(num1)==0: return num2
        if len(num2)==0: return num1
        
        if len(num1) < len(num2):
            num1, num2 = num2, num1
        
        s =''
        carry = 0
        j=len(num2)-1
        for i in range(len(num1)-1,-1,-1):
            x = self.dict_num[num1[i]] + carry
            if j>=0:
                x = x + self.dict_num[num2[j]]
            s = str(x%10) + s
            carry = x/10
            j=j-1
        if carry>0: s = str(carry) + s
        return s
            
```
