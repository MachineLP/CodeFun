```
You may assume that the intervals were initially sorted according to their start times.

Example 1:

Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]
Example 2:

Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
Output: [[1,2],[3,10],[12,16]]
Explanation: Because the new interval [4,8] overlaps with [3,5],[6,7],[8,10].
```

## 换一种思路可能不是最优的，但是可以最快的：转换为跟56一样的问题。
```python
# Definition for an interval.
# class Interval(object):
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution(object):
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[Interval]
        :type newInterval: Interval
        :rtype: List[Interval]
        """
        out = []
        intervals.append( newInterval )
        for per_interval in sorted(intervals, key=lambda i:i.start):
            if out and per_interval.start <= out[-1].end:
                out[-1].end = max(per_interval.end, out[-1].end)
            else:
                out.append( per_interval )
            
        return out
```
