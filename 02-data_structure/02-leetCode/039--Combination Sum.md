
```
Given a set of candidate numbers (candidates) (without duplicates) and a target number (target), find all unique combinations in candidates where the candidate numbers sums to target.

The same repeated number may be chosen from candidates unlimited number of times.

Note:

All numbers (including target) will be positive integers.
The solution set must not contain duplicate combinations.
Example 1:

Input: candidates = [2,3,6,7], target = 7,
A solution set is:
[
  [7],
  [2,2,3]
]
Example 2:

Input: candidates = [2,3,5], target = 8,
A solution set is:
[
  [2,2,2,2],
  [2,3,3],
  [3,5]
]
```


```python
class Solution(object):
    def __init__(self):
        self.one_path = []
        self.path = []
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        if len(candidates) <=0:
            return []
        start = 0
        self.combination_sum_helper(candidates, start, target)
        return self.path
    
    def combination_sum_helper(self, candidates, start, target):
        if target<0:
            return
        if target == 0:
            self.path.append(self.one_path[:])
            return 
        '''
        for i in range( start,len(candidates) ):
            if i>start and candidates[i]==candidates[i-1]:
                continue
            self.one_path.append(candidates[i])
            self.combination_sum_helper(candidates, i, target-candidates[i])
            self.one_path.pop()'''
        for i in range(len(candidates) ):
            if i>start and candidates[i]==candidates[i-1]:
                continue
            self.one_path.append(candidates[i])
            self.combination_sum_helper(candidates[i:], i, target-candidates[i])
            self.one_path.pop()
            
```
