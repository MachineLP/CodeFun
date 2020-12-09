
![image](https://assets.leetcode.com/uploads/2018/10/22/rainwatertrap.png)
```
Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.


The above elevation map is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped. Thanks Marcos for contributing this image!

Example:

Input: [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
```

```python
class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        n = len(height)
        res = 0
        max_index = 0
        max_value = 0
        for i in range(n):
            if height[i] > max_value:
                max_value = height[i]
                max_index = i
        
        pre_value = 0
        for i in range(max_index):
            if height[i] > pre_value:
                pre_value = height[i]
            res += pre_value - height[i]
        
        pre_value = 0
        for i in range(n-1,max_index,-1):
            if height[i] > pre_value:
                pre_value = height[i]
            res += pre_value - height[i]
            
        return res

```
