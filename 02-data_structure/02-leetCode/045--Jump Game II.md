
```
Given an array of non-negative integers, you are initially positioned at the first index of the array.

Each element in the array represents your maximum jump length at that position.

Your goal is to reach the last index in the minimum number of jumps.

Example:

Input: [2,3,1,1,4]
Output: 2
Explanation: The minimum number of jumps to reach the last index is 2.
    Jump 1 step from index 0 to 1, then 3 steps to the last index.
Note:

You can assume that you can always reach the last index.
```


```python
class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) <2:
            return 0
        
        current_max_index = nums[0]
        pre_max_index = nums[0]
        jump_min = 1
        
        for i in range(len(nums)):
            if i > current_max_index:   # 当无法再向前移动时，进行跳跃
                jump_min +=1
                current_max_index = pre_max_index
                
            if pre_max_index < nums[i]+i:
                pre_max_index = nums[i]+i
        return jump_min
```
