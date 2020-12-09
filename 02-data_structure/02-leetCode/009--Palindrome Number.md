
```
Determine whether an integer is a palindrome. An integer is a palindrome when it reads the same backward as forward.

Example 1:

Input: 121
Output: true
Example 2:

Input: -121
Output: false
Explanation: From left to right, it reads -121. From right to left, it becomes 121-. Therefore it is not a palindrome.
Example 3:

Input: 10
Output: false
Explanation: Reads 01 from right to left. Therefore it is not a palindrome.
```

```python
class Solution(object):
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        '''
        x = str(x)
        len_x = len(x)
        flg = True
        for i in range( len_x/2 ):
            if x[i] != x[len_x-1-i]:
                flg = False
                break
        return flg'''
        #return str(x) == str(x)[::-1] 
        y = 0
        a = x
        while a>0:
            y  = y * 10 + a%10
            a = a/10
        print (y)
        return y == x 
```
