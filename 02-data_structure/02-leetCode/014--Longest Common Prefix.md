
```
Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string "".

Example 1:

Input: ["flower","flow","flight"]
Output: "fl"
Example 2:

Input: ["dog","racecar","car"]
Output: ""
Explanation: There is no common prefix among the input strings.
Note:

All given inputs are in lowercase letters a-z.

```


```python
class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if strs == None or len(strs) == 0:
            return ''
        
        for i in range( len(strs[0]) ):
            c = strs[0][i]
            for j in range( len(strs) ) :
                if i == len(strs[j]) or strs[j][i] != c:
                    return strs[0][0:i]
                
        return strs[0]
```
