```
题目描述

求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。

解题思路

本题为数学问题，对一个数n来讲，从最低位开始遍历，假设当前位的数字为cur，该数字左边的数大小为left，右边的数大小为right。此时对该位来讲，出现1要分三种情况讨论：

cur = 0：若要该位为1，只有左边的数小于left，即左边0至left-1，共left种，对左边每个数而言，right有num位，就有10^num（x）种情况

cur = 1：除了上述情况，还需要加上左边为left，则该情况下，右边可以出现的数为0-right，即right+1中，所以在第一种的情况下还需要加上right+1

cur > 1：与第一种不同的是，左边可取的范围为0-left，left+1种，左边每种情况，右边就有10^num（x）种。

```

```C++
class Solution {
public:
    int NumberOf1Between1AndN_Solution(int n)
    {
        int left = n, right = 0, cur, ans = 0, x = 1;
        while(left){
            cur = left % 10;
            left /= 10;
            if(cur == 0)
                ans += left * x;
            else if(cur == 1)
                ans += left * x + right + 1;
            else
                ans += left * x + x;
            right += cur * x;
            x *= 10;
        }
        return ans;
    }
};


```

```python
class Solution:
    def NumberOf1Between1AndN_Solution(self, n):
        # write code here
        left, cur, right, x, ans = n, 0, 0, 1, 0
        while left:
            cur = left % 10
            left /= 10
            if cur == 0:
                ans += left * x
            elif cur == 1:
                ans += left * x + right + 1
            else:
                ans += left * x + x
            right += x * cur
            x *= 10
        return ans


```
