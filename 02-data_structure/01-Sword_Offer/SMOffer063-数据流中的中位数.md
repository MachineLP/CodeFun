```
题目描述

如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。我们使用Insert()方法读取数据流，使用GetMedian()方法获取当前读取数据的中位数。

解题思路

将整个数据分为两个部分，左边部分为数据中较小的一半，即右边部分为数据中较大的一半，若总个数为奇数，左边部分放k+1个，右边放k个。考虑到中途需要将两部分数据调动到另一部分中，我们需要调动的数据，左边一定为其最大的数，右边一定为其最小的数，同时在计算中位数，若总数为奇数，则中位数为左边最大的数，总数为偶数，则中位数为左边最大的数和右边最小的数的平均。从这里可以看出来，我们需要的就是以尽可能快的方法取到左边的最大的数，右边最小的数，我们想到堆排序，可以使得取数的复杂度为O(1)，插入数据的复杂度为O(logn)。

具体做法：将数分为两个部分，左边最大堆，右边最小堆存放。在插入数据的过程中保证右边的数据都比左边大，且两边的数据数量相同或者左边比右边多1。

```


```C++
class Solution {
public:
    void Insert(int num)
    {
        if(((max.size() + min.size()) & 1) == 0){
            if(max.size() > 0 && num < max[0]){
                max.push_back(num);
                push_heap(max.begin(), max.end(), less<int>());
                num = max[0];
                pop_heap(max.begin(), max.end(), less<int>());
                max.pop_back();
            }
            min.push_back(num);
            push_heap(min.begin(), min.end(), greater<int>());
        }
        else{
            if(min.size() > 0 && num > min[0]){
                min.push_back(num);
                push_heap(min.begin(), min.end(), greater<int>());
                num = min[0];
                pop_heap(min.begin(), min.end(), greater<int>());
                min.pop_back();
            }
            max.push_back(num);
            push_heap(max.begin(), max.end(), less<int>());
        }
    }
 
    double GetMedian()
    { 
        int size = min.size() + max.size();
        if(size == 0)
            return 0;
        if((size & 1) == 0)
            return (min[0] + max[0]) / 2.0;
        else
            return min[0];
    }
private:
    vector<int>min, max;
};


```


```python
import heapq
class Solution:
    def __init__(self):
        self.left_num = []
        self.right_num = []
    def Insert(self, num):
        # write code here
        size = len(self.left_num) + len(self.right_num)
        if size & 1:
            num = -heapq.heappushpop(self.left_num, -num)
            heapq.heappush(self.right_num, num)
        else:
            num = heapq.heappushpop(self.right_num, num)
            heapq.heappush(self.left_num, -num)
    def GetMedian(self, num):
        # write code here
        size = len(self.left_num) + len(self.right_num)
        if size & 1:
            return -1.0 * self.left_num[0]
        else:
            return (self.right_num[0] - self.left_num[0]) / 2.0


```
