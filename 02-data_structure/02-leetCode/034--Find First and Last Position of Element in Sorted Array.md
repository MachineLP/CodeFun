
```
Given an array of integers nums sorted in ascending order, find the starting and ending position of a given target value.

Your algorithm's runtime complexity must be in the order of O(log n).

If the target is not found in the array, return [-1, -1].

Example 1:

Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]
Example 2:

Input: nums = [5,7,7,8,8,10], target = 6
Output: [-1,-1]
```


```python
class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
               
        n = len(nums)
        res0 = self.binary_search1(nums, n, target)
        # res1 = self.binary_search2(nums, n, target)
        res = res0
        print (res)
        for per_num in nums[res0+1:]:
            if per_num == target:
                res = res+1
            else:
                break
                
        return [res0, res]
    
    # 返回第一个等于给定值的元素
    def binary_search1(self, nums, n, target):
        low = 0
        high = n-1
        while low <= high:
            mid = low + (high-low) // 2
            if nums[mid] < target:
                low = mid + 1
            elif nums[mid] > target:
                high = mid - 1
            else:
                if mid==0 or nums[mid-1]!=target: return mid
                else: high = mid - 1
        return -1
    
    # 返回最后一个等于给定值的元素
    def binary_search2(self, nums, n, target):
        low = 0
        high = n-1
        while low <= high:
            mid = low + (high-low) // 2
            if nums[mid] < target:
                low = mid + 1
            elif nums[mid] > target:
                high = mid - 1
            else:
                if mid==n-1 or nums[mid+1]!=target: return mid
                else: low = mid + 1
        return -1
    
```
