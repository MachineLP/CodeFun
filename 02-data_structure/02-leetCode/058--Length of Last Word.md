```
Given a string s consists of upper/lower-case alphabets and empty space characters ' ', return the length of last word in the string.

If the last word does not exist, return 0.

Note: A word is defined as a character sequence consists of non-space characters only.

Example:

Input: "Hello World"
Output: 5
```

## 任何一个简单的功能都有很多坑，经验的价值。
```python
class Solution(object):
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """ 
        if ' ' not in s:
            return len(s)
        flag = True
        res_list = []
        str_list = s.split(' ')
        for index, per_s in enumerate(str_list):
            if len(per_s) == 0:
                if index == len(str_list)-1 and flag:
                    return 0
                continue
            else:
                res_list.append( per_s )
                flag = False
        return len(res_list[-1])
```

