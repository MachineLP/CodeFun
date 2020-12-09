
```
Given a string containing just the characters '(' and ')', find the length of the longest valid (well-formed) parentheses substring.

Example 1:

Input: "(()"
Output: 2
Explanation: The longest valid parentheses substring is "()"
Example 2:

Input: ")()())"
Output: 4
Explanation: The longest valid parentheses substring is "()()"
```


```python
class Solution(object):
    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        
        max_len = 0
        last_error = -1
        stack = []
        for i in range( len(s) ):
            if s[i] == '(':
                stack.append(i)
            elif s[i] == ')':
                if len(stack) > 0:
                    stack.pop()
                    if len(stack) == 0:
                        len_i = i - last_error
                    else:
                        len_i = i - stack[-1]
                    if (len_i > max_len):
                        max_len = len_i
                else:
                    last_error = i
                    
        return max_len
                
```
