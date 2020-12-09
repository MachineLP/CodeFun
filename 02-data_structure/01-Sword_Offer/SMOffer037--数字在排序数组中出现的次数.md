```
题目描述

统计一个数字在排序数组中出现的次数。

解题思路

找到该数字在数组中第一次和最后一次出现的下标，即可求得次数。为提高时间效率，利用二分法查找。

相比以往的查找某个元素的位置不同的是：

对firstpos来讲，当当前的mid处的值等于k，且mid为数组第一个元素或者，mid处的前一个值不为k，则mid为firstpos

对endpos来讲，当当前的mid处的值等于k，且mid为数组最后一个元素或者，mid处的后一个值不为k，则mid为endpos

```


```C++
class Solution {
public:
    int GetNumberOfK(vector<int> data ,int k) {
        if(data.size() == 0 || k < data[0] || k > data[data.size() - 1])
            return 0;
        int cnt = 0, low = 0, high = data.size() - 1, lastpos, firstpos = -1;
        while(low <= high){
            int mid = (high + low) >> 1;
            if(data[mid] == k){
                if(mid > 0 && data[mid-1] != k || mid == 0){
                    firstpos = mid;
                    break;
                }
                else
                    high = mid - 1;
            }
            else if(data[mid] > k)
                high = mid - 1;
            else
                low = mid + 1;
        }
        if(firstpos == -1)
            return 0;
        low = 0;
        high = data.size() - 1;
        while(low <= high){
            int mid = (high + low) >> 1;
            if(data[mid] == k){
                if(mid == data.size() - 1 || (mid < data.size() - 1 && data[mid+1] != k)){
                    lastpos = mid;
                    break;
                }
                else
                    low = mid + 1;
            }
            else if(data[mid] > k)
                high = mid - 1;
            else
                low = mid + 1;
        }
        return lastpos - firstpos + 1;
    }
};


```


```python
class Solution:
    def GetNumberOfK(self, data, k):
        # write code here
        if len(data) == 0 or data[0] > k or data[-1] < k:
            return 0
        firstpos, lastpos = -1, -1
        low, high = 0, len(data) - 1
        while low <= high:
            mid = (low + high) / 2
            if data[mid] == k:
                if mid == 0 or data[mid-1] != k:
                    firstpos = mid
                    break
                else:
                    high = mid - 1
            elif data[mid] > k:
                high = mid - 1
            else:
                low = mid + 1
        if firstpos == -1:
            return 0
        low, high = 0, len(data) - 1
        while low <= high:
            mid = (low + high) / 2
            if data[mid] == k:
                if mid == len(data) - 1 or data[mid+1] != k:
                    lastpos = mid
                    break
                else:
                    low = mid + 1
            elif data[mid] > k:
                high = mid - 1
            else:
                low = mid + 1
        return lastpos - firstpos + 1


```
