
```
Given n non-negative integers a1, a2, ..., an , where each represents a point at coordinate (i, ai). n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). Find two lines, which together with x-axis forms a container, such that the container contains the most water.

Note: You may not slant the container and n is at least 2.

Example:

Input: [1,8,6,2,5,4,8,3,7]
Output: 49
```

```python
class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        
        length = len(height)
        l=0
        r=length-1
        max_area=0
        
        while l<r:
            diff=r-l # calculate x length
            if height[l]<height[r]: # this is the max area that height[l] can form, so, move l                                       # pointer to the next one 
                max_area=max(diff*height[l],max_area)
                l+=1
            else: # this means height[l] can form bigger area, so move r not l
                max_area=max(diff*height[r],max_area)
                r-=1
        return max_area
```


```python
# 进一步优化一下
class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        
        # 进一步优化：
        length = len(height)
        l=0
        r=length-1
        max_area=0
        
        while l<r:
            diff=r-l # calculate x length
            if height[l]<height[r]: # this is the max area that height[l] can form, so, move l                                       # pointer to the next one 
                max_area=max(diff*height[l],max_area)
                while l<r and height[l]>=height[l+1]:
                    l+=1
                l+=1
            else: # this means height[l] can form bigger area, so move r not l
                max_area=max(diff*height[r],max_area)
                while l<r and height[r]>=height[r-1]:
                    r-=1
                    #print (r)
                r-=1
                #print ('>>>', r)
        return max_area
        
```
