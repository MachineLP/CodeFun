
```
Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sums to target.

Each number in candidates may only be used once in the combination.

Note:

All numbers (including target) will be positive integers.
The solution set must not contain duplicate combinations.
Example 1:

Input: candidates = [10,1,2,7,6,1,5], target = 8,
A solution set is:
[
  [1, 7],
  [1, 2, 5],
  [2, 6],
  [1, 1, 6]
]
Example 2:

Input: candidates = [2,5,2,1,2], target = 5,
A solution set is:
[
  [1,2,2],
  [5]
]
```


```python
class Solution(object):
    def __init__(self):
        self.one_path = []
        self.path = []
    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        if len(candidates) <=0:
            return []
        candidates = sorted(candidates)
        start = 0
        self.combination_sum_helper(candidates, start, target)
        return self.path
    
    def combination_sum_helper(self, candidates, start, target):
        if target<0:
            return
        if target == 0:
            if self.one_path[:] not in self.path:
                self.path.append(self.one_path[:])
                return 
        '''
        for i in range( start,len(candidates) ):
            self.one_path.append(candidates[i])
            self.combination_sum_helper(candidates, i+1, target-candidates[i])
            self.one_path.pop()
        '''
        for i in range(len(candidates) ):
            self.one_path.append(candidates[i])
            self.combination_sum_helper(candidates[i+1:], i, target-candidates[i])
            self.one_path.pop()
    
    # 贪心算法
    # 动态规划算法
    
                
```

```python
import numpy as np
class Solution(object):
    def __init__(self):
        self.one_path = []
        self.path = []
    
    # 回溯法
    '''
    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        if len(candidates) <=0:
            return []
        candidates = sorted(candidates)
        start = 0
        self.combination_sum_helper(candidates, start, target)
        return self.path
    
    def combination_sum_helper(self, candidates, start, target):
        if target<0:
            return
        if target == 0:
            if self.one_path[:] not in self.path:
                self.path.append(self.one_path[:])
                return 
    '''
    #    '''
    #    for i in range( start,len(candidates) ):
    #        self.one_path.append(candidates[i])
    #        self.combination_sum_helper(candidates, i+1, target-candidates[i])
    #        self.one_path.pop()
    #    '''
    '''
        for i in range(len(candidates) ):
            self.one_path.append(candidates[i])
            self.combination_sum_helper(candidates[i+1:], i, target-candidates[i])
            self.one_path.pop()
    '''
    # 贪心算法
    def combinationSum2(self, candidates, target):
        candidates = sorted(candidates)
        self.mem = np.zeros( (len(candidates)+1, target+1) )
        self.candidates = candidates
        self.target = target
        self.f(0,0)
        return self.path

    def f(self, i, cw):
        if cw == self.target:
            if self.one_path[:] not in self.path:
                self.path.append(self.one_path[:])
            return 
        if i>=len(self.candidates): 
            return
        #if self.mem[i][cw]: return 
        #self.mem[i][cw] = 1
        self.f(i+1, cw)
        if cw+self.candidates[i]<=self.target:
            self.one_path.append(self.candidates[i])
            self.f(i+1, cw+self.candidates[i])
            self.one_path.pop()
    
    
    # 动态规划算法
    # 。。。。。。
```
