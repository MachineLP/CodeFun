```
题目描述

小明很喜欢数学,有一天他在做数学作业时,要求计算出9~16的和,他马上就写出了正确答案是100。但是他并不满足于此,他在想究竟有多少种连续的正数序列的和为100(至少包括两个数)。没多久,他就得到另一组连续正数和为100的序列:18,19,20,21,22。现在把问题交给你,你能不能也很快的找出所有和为S的连续正数序列? Good Luck!

输出描述:

输出所有和为S的连续正数序列。序列内按照从小至大的顺序，序列间按照开始数字从小到大的顺序
解题思路

双指针法，设置两个指针low，high分别表示序列的最小值和最大值。low初始为1，high初始为2。若序列和大于sum，则low向前移，若小于，则high后移，最终当low到达(sum + 1) >> 1停止循环。

```


```C++
class Solution {
public:
    vector<vector<int> > FindContinuousSequence(int sum) {
        vector<vector<int>>ans;
        if(sum < 3)
            return ans;
        int low = 1, high = 2, mid = (sum + 1) >> 1, cur = 3;
        vector<int>tempans;
        tempans.push_back(1);
        tempans.push_back(2);
        while(low < mid){
            if(cur == sum)
                ans.push_back(tempans);
            while(cur > sum && low < mid){
                cur -= low;
                tempans.erase(tempans.begin());
                ++ low;
                if(cur == sum)
                    ans.push_back(tempans);
            }
            ++ high;
            cur += high;
            tempans.push_back(high);
        }
        return ans;
    }
};


```

```python
class Solution:
    def FindContinuousSequence(self, tsum):
        # write code here
        ans, tempans = [],[1, 2]
        low, high, mid, tempsum = 1, 3, (tsum + 1) >> 1, 3
        if tsum < 3:
            return ans
        while low < mid:
            if tempsum == tsum:
                ans.append(tempans[:])
            if tempsum > tsum:
                tempsum -= low
                low += 1
                tempans.pop(0)
            else:
                tempsum += high
                tempans.append(high)
                high += 1
        return ans


```
