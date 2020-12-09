```
Given a non-empty array of digits representing a non-negative integer, plus one to the integer.

The digits are stored such that the most significant digit is at the head of the list, and each element in the array contain a single digit.

You may assume the integer does not contain any leading zero, except the number 0 itself.

Example 1:

Input: [1,2,3]
Output: [1,2,4]
Explanation: The array represents the integer 123.
Example 2:

Input: [4,3,2,1]
Output: [4,3,2,2]
Explanation: The array represents the integer 4321.
```

### 思路不一样解法完全不一样。
```python
class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        carry =1
        for i in range(len(digits)-1,-1,-1):
            if carry>0:
                digits[i]+=carry
                carry = digits[i]/10
                digits[i] = digits[i]%10
            else:
                return digits
        return [1]+digits if carry>0 else digits
        '''
        s=''
        for i in digits:
            s += str(i)
        k=[]
        for j in str(int(s)+1):
            k.append(int(j))
        return(k)
        '''
```
