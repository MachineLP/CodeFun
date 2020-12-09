
```
Given an array of strings, group anagrams together.

Example:

Input: ["eat", "tea", "tan", "ate", "nat", "bat"],
Output:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]
Note:

All inputs will be in lowercase.
The order of your output does not matter.
```


```python
class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        solution = []
        ana_dict = {} 
        for x in strs:
            a = ''.join(sorted(x))
            # print (a)
            if a in ana_dict:
                ana_dict[a].append(x)
            else:
                ana_dict[a] = [x]
                
        for key, values in ana_dict.items():
            solution.append(values)    
            
        # print (solution)
        return solution
```
