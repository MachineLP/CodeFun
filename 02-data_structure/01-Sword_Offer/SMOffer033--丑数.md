```
题目描述

把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。

解题思路

最直观的方法是逐一进行判断，直到找到第N个丑数。很明显这种方法复杂度很大，进一步考虑，丑数是只包含质因子2、3和5的数，另一方面如果仅仅由质因子2、3和5相乘得到的数一定不包含其他的质因子，也就是为丑数。

我们创建一个数组ugly用来保存丑数，数组中的丑数已经排好序，接下来如何生成下一个按顺序的丑数，该丑数一定为前面某一个丑数*2或*3或*5得到的，用三个指针记录哪个丑数*2或*3或*5可以超过当前最大的丑数。

```


``C++
class Solution {
public:
    int GetUglyNumber_Solution(int index) {
        vector<int>ugly(index + 1);
        ugly[0] = 1;
        int n2 = 0, n3 = 0, n5 = 0;
        for(int i = 1; i <= index; ++ i){
            ugly[i] = min(ugly[n2] * 2,ugly[n3] * 3);
            ugly[i] = min(ugly[i],ugly[n5] * 5);
            if(ugly[i] == ugly[n2] * 2)
                ++ n2;
            if(ugly[i] == ugly[n3] * 3)
                ++ n3;
            if(ugly[i] == ugly[n5] * 5)
                ++ n5;
        }
        return ugly[index - 1];
    }
};


```


```python
class Solution:
    def GetUglyNumber_Solution(self, index):
        # write code here
        n2, n3, n5 = 1, 1, 1
        ugly = [0,1]
        for i in range(0, index):
            num = min(min(ugly[n2]*2, ugly[n3]*3), ugly[n5]*5)
            if num == ugly[n2]*2:
                n2 += 1
            if num == ugly[n3]*3:
                n3 += 1
            if num == ugly[n5]*5:
                n5 += 1
            ugly.append(num)
        return ugly[index]


```
