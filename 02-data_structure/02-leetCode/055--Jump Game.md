
```
Given an array of non-negative integers, you are initially positioned at the first index of the array.

Each element in the array represents your maximum jump length at that position.

Determine if you are able to reach the last index.
Example 1:

Input: [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
Example 2:

Input: [3,2,1,0,4]
Output: false
Explanation: You will always arrive at index 3 no matter what. Its maximum
             jump length is 0, which makes it impossible to reach the last index.
```


```python
class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        
        if len(nums) == 0:
            return True 
        
        last_reach = 0
        for i, x in enumerate(nums):
            
            # i cannot be reached by any previous index 
            # [3,2,1,0,4], i = 4, last = 3
            if i > last_reach:
                break 
            
            # update last reachable element if i + x reaches further 
            last_reach = max(last_reach, i + x) 
            
            # if the last element can be reached 
            # [2,3,1,1,4], i = 1, x = 3, last = 4 = n - 1
            if last_reach >= len(nums) - 1:
                return True
        
        return False
```
