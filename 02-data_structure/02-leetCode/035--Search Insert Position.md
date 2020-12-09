
```
Given a sorted array and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

You may assume no duplicates in the array.

Example 1:

Input: [1,3,5,6], 5
Output: 2
Example 2:

Input: [1,3,5,6], 2
Output: 1
Example 3:

Input: [1,3,5,6], 7
Output: 4
Example 4:

Input: [1,3,5,6], 0
Output: 0
```


```python
class Solution(object):
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        n = len(nums)
        return self.binary_search1(nums, n, target)+1
        
    # 返回最后一个小于给定值的元素
    def binary_search1(self, nums, n, target):
        low = 0
        high = n-1
        while low <= high:
            mid = low + (high-low) // 2
            if nums[mid] < target:
                if mid==n-1 or nums[mid+1] >=target: return mid
                else: low = mid + 1
            else:
                high = mid - 1
        return -1
```
