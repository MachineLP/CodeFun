
```
Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e., [0,1,2,4,5,6,7] might become [4,5,6,7,0,1,2]).

You are given a target value to search. If found in the array return its index, otherwise return -1.

You may assume no duplicate exists in the array.

Your algorithm's runtime complexity must be in the order of O(log n).

Example 1:

Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
Example 2:

Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1
```


```python
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        
        n = len(nums)
        if n<=0: return -1
        if n==1:
            if nums[0]==target:
                return 0
            else:
                return -1
        
        if nums[0]<nums[-1]:
            return self.binary_search2(nums, 0, n-1, target)
        else:
            return self.rotate_search(nums, 0, n-1, target);
    
    def binary_search1(self, nums, target):
        low = 0
        high = len(nums)-1
        while low <= high:
            mid = low + (high-low) // 2
            if nums[mid] == target:
                return mid
            if nums[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
        return -1
    
    def binary_search2(self, nums, low, high, target):
        if low>high: return -1
        mid = low + (high-low)//2
        if nums[mid]==target:
            return mid
        if nums[mid]>target:
            return self.binary_search2(nums, low, mid-1, target)
        else:
            return self.binary_search2(nums, mid+1, high, target)
    
    def rotate_search(self, nums, low, high, target):
        if low>high: return -1
        if low==high:
            if nums[low]==target:
                return low
            else:
                return -1
        mid = low + (high-low)//2
        if nums[mid]==target:
            return mid
        if nums[low]<nums[mid] and target>=nums[low] and target<nums[mid]:
            return self.binary_search2(nums, low, mid-1, target)
        if nums[mid]<nums[high] and target>nums[mid] and target<=nums[high]:
            return self.binary_search2(nums, mid+1, high, target)
        if nums[low]>nums[mid]:
            return self.rotate_search(nums, low, mid-1, target)
        if nums[mid]>nums[high]:
            return self.rotate_search(nums, mid+1, high, target)
        return -1
    

```
