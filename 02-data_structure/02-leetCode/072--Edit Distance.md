```
Given two words word1 and word2, find the minimum number of operations required to convert word1 to word2.

You have the following 3 operations permitted on a word:

Insert a character
Delete a character
Replace a character
Example 1:

Input: word1 = "horse", word2 = "ros"
Output: 3
Explanation: 
horse -> rorse (replace 'h' with 'r')
rorse -> rose (remove 'r')
rose -> ros (remove 'e')
Example 2:

Input: word1 = "intention", word2 = "execution"
Output: 5
Explanation: 
intention -> inention (remove 't')
inention -> enention (replace 'i' with 'e')
enention -> exention (replace 'n' with 'x')
exention -> exection (replace 'n' with 'c')
exection -> execution (insert 'u')
```


```python
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
    
        self.cache = {}
        return self.computeLevenshtein(word1, len(word1), word2, len(word2))
    
    def computeLevenshtein(self, s1, i, s2, j):
        if i <= 0:
            return j

        if j <= 0:
            return i

        s1 = s1[:i]
        s2 = s2[:j]
        if self.cache.get(s1+s2):
            return self.cache[s1+s2]

        if s1[i-1] == s2[j-1]:
            dist = self.computeLevenshtein(s1, i - 1, s2, j - 1)
            self.cache[s1+s2] = dist
            return dist

        else:
            dist = min(
                self.computeLevenshtein(s1, i - 1, s2, j - 1),
                self.computeLevenshtein(s1, i - 1, s2, j),
                self.computeLevenshtein(s1, i, s2, j - 1))
            self.cache[s1+s2] = dist + 1
            return dist + 1
        
        '''
        if not word1 and not word2:
            return 0
        if not word1:
            return len(word2)
        if not word2:
            return len(word1)
        if word1[0] == word2[0]:
            return self.minDistance(word1[1:], word2[1:])
        insert = 1 + self.minDistance(word1, word2[1:])
        delete = 1 + self.minDistance(word1[1:], word2)
        replace = 1 + self.minDistance(word1[1:], word2[1:])
        return min(insert, replace, delete)
        '''
```

