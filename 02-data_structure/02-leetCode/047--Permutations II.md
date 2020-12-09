
```
Given a collection of numbers that might contain duplicates, return all possible unique permutations.

Example:

Input: [1,1,2]
Output:
[
  [1,1,2],
  [1,2,1],
  [2,1,1]
]
```


```python
class Solution(object):
    import copy
    def __init__(self):
        self.result = []
        self.per_result = []
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if len(nums) == 0:
            if self.per_result not in self.result:
                self.result.append(self.per_result[:])
            # return 
        
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            self.per_result.append(nums[i])
            rest_nums = copy.deepcopy( nums )
            rest_nums.pop(i)
            self.permuteUnique(rest_nums)
            self.per_result.pop()
        return self.result
    
    '''
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        result = []
        self.permutation(nums, [],result)
        return result

    def permutation(self, numbers, curr, result):
        if len(numbers) == 0:
            result.append(curr)

        for i in range(len(numbers)):
            if i > 0 and numbers[i] == numbers[i-1]:
                continue
            self.permutation(numbers[0:i]+numbers[i+1:], curr + [numbers[i]], result)
    '''
```
