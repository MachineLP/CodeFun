
```
Implement next permutation, which rearranges numbers into the lexicographically next greater permutation of numbers.

If such arrangement is not possible, it must rearrange it as the lowest possible order (ie, sorted in ascending order).

The replacement must be in-place and use only constant extra memory.

Here are some examples. Inputs are in the left-hand column and its corresponding outputs are in the right-hand column.

1,2,3 → 1,3,2
3,2,1 → 1,2,3
1,1,5 → 1,5,1
```


```python
class Solution(object):
    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        nums_length = len(nums)
        if (nums_length <1 ):
            return 
        
        for i in range(nums_length-1, 0, -1):
            if nums[i-1] < nums[i]:
                j = nums_length-1
                while nums[i-1] >= nums[j]:
                    j -= 1
                nums[i-1], nums[j] = nums[j], nums[i-1]
                nums[i:] = list( reversed( nums[i:] ) )
                return 
            if i==1:
                nums[:] = list( reversed( nums[:] ) )
                return 
```
