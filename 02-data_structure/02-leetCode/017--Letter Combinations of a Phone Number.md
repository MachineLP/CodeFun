
```
Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent.

A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.



Example:

Input: "23"
Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
```


```python
class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        # map与reduce用法: https://www.cnblogs.com/weiman3389/p/6047095.html
        
        if len(digits)==0:
            return []
        d= {"2":["a", "b", "c"], "3":["d", "e", "f"], "4":["g", "h", "i"], "5":["j", "k", "l"], "6":["m", "n", "o"], "7":["p", "q", "r", "s"], "8":["t", "u", "v"], "9":["w", "x", "y", "z"]}
        s= list(map((lambda x: d[x]),list(digits)))
        print (s)
        
        def digitComb(X,Y):
            li=[]
            for x in X:
                for y in Y:
                    li.append(x+y)
            return li
        
        return reduce((lambda x,y: digitComb(x,y)), s)
```

