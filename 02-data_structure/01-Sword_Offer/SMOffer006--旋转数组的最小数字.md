```
题目描述

把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。 输入一个非减排序的数组的一个旋转，输出旋转数组的最小元素。 例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。

NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。

解题思路

一般而言，非排序的数组上的操作，面试官期待的是O(n)的解法，而排序数组上的操作，则是O(logn)的解法。

本题用二分法，可以注意到，旋转数组可以分为两个递增数组，设两个指针，初始的时候分别第一个和最后一个元素，即分别位于前面一个递增数组和后面一个递增数组。接下来我们要做的是，保持这种状态，即第一个指针在前面一个递增数组，第二个指针在后面一个递增数组，不断缩小这个范围，最终找到两个递增数组的分界线，即数组中最小的数字。接下的问题就是如何保持两个指针的这种状态。

首先明确一下，两指针之间的元素一定要么大于第一个指针的元素，要么小于第二个指针的元素。

当中间元素大于第一个指针的元素，一定有分界线在第二个指针和中间元素之间，此时修改第一个指针

当中间元素小于第二个指针的元素，一定有分界线在第一个指针和中间元素之间，此时修改第二个指针

```

```C++
class Solution {
public:
    int minNumberInRotateArray(vector<int> rotateArray) {
        int low = 0, high = rotateArray.size() - 1, mid = 0;
        while(rotateArray[low] >= rotateArray[high]){
            if(high - low == 1){
                mid = high;
                break;
            }
            mid = (high + low) / 2;
            if(rotateArray[mid] >= rotateArray[low])
                low = mid;
            if(rotateArray[mid] <= rotateArray[high])
                high = mid;
        }
        return rotateArray[mid];
    }
};


```

```python
# -*- coding:utf-8 -*-
class Solution:
    def minNumberInRotateArray(self, rotateArray):
        # write code here
        low = 0
        high = len(rotateArray)-1
        mid = 0
        if high == -1:
            return 0
        while rotateArray[low] >= rotateArray[high]:
            if high - low == 1:
                mid = high
                break
            mid = (high + low) / 2
            if rotateArray[mid] >= rotateArray[low]:
                low = mid
            elif rotateArray[mid] <= rotateArray[high]:
                high = mid
        return rotateArray[mid]


```
