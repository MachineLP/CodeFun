
```
Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order.
Note that an empty string is also considered valid.

Example 1:

Input: "()"
Output: true
Example 2:

Input: "()[]{}"
Output: true
Example 3:

Input: "(]"
Output: false
Example 4:

Input: "([)]"
Output: false
Example 5:

Input: "{[]}"
Output: true
```

```python
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        sta = []
        
        for per_s in s:
            if per_s=='(' or per_s=='[' or per_s=='{':
                sta.append(per_s)
            elif per_s==')' or per_s==']' or per_s=='}':
                if sta == []: return False
                temp_s = sta.pop()
                if per_s==')' and temp_s=='(' or per_s==']' and temp_s=='[' or per_s=='}' and temp_s=='{':
                    continue
                else:
                    return False
                
            else:
                return False
        
        return sta == []
```
