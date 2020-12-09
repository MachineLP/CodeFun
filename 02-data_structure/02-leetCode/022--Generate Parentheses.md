
```
Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

For example, given n = 3, a solution set is:

[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]
```

```python
class Solution(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        res = []
        s = ''
        self.generator(res, n, n, s)
        return res
    
    def generator(self, res, left, right, s):
        
        if left==0 and right==0:
            res.append(s)
            return 
        
        if (left>0):
            self.generator( res, left-1, right, s+'(' )
        if (right>0 and right>left):
            self.generator( res, left, right-1, s+')' )

```
