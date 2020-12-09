
```
Given an array nums of n integers and an integer target, find three integers in nums such that the sum is closest to target. Return the sum of the three integers. You may assume that each input would have exactly one solution.

Example:

Given array nums = [-1, 2, 1, -4], and target = 1.

The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).
```


```python
# 基于015的修改
class Solution(object):
    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        res = 0
        nums = sorted(nums)
        nums_length = len(nums)
        closest_data = 1000000000

        for i in range(nums_length-2):

            if i>0 and nums[i-1]==nums[i]:
                continue
            a = nums[i]
            low = i + 1
            high = nums_length - 1
            while (low < high):
                b = nums[low]
                c = nums[high]
                if a+b+c==target:
                    res = a+b+c
                    return res
                    '''
                    res.append([a,b,c])
                    while(low<nums_length-1 and nums[low]==nums[low+1]):
                        low += 1
                    while(high>0 and nums[high]==nums[high-1]):
                        high -= 1
                    low += 1
                    high -= 1
                    '''
                elif (a+b+c)>target:
                    while(high>0 and nums[high]==nums[high-1]):
                        high -= 1
                    high -= 1
                    temp = (a+b+c) - target
                    if  temp < closest_data:
                        closest_data = temp
                        res = (a+b+c)
                else:
                    while(low<nums_length-1 and nums[low]==nums[low+1]):
                        low += 1
                    low += 1
                    temp = target - (a+b+c)
                    if  temp < closest_data:
                        closest_data = temp
                        res = (a+b+c)
        return res

```

