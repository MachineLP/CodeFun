
```
Given a collection of distinct integers, return all possible permutations.

Example:

Input: [1,2,3]
Output:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]
```


```python
class Solution(object):
    import copy
    def __init__(self):
        self.result = []
        self.per_result = []
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        
        if len(nums) == 0:
            self.result.append(self.per_result[:])
            # return 
        
        for i in range(len(nums)):
            self.per_result.append(nums[i])
            rest_nums = copy.deepcopy( nums )
            rest_nums.pop(i)
            self.permute(rest_nums)
            self.per_result.pop()
        return self.result
        
```
