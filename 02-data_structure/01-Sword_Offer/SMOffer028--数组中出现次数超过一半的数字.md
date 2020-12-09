```
题目描述

数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。

解题思路

本题最简单快捷的方法是贪心法，巧妙利用结果的特点，如果有一个数字出现次数超过数组长度的一半，则它的出现次数比其他所有数字出现的次数之和还要多。因此在遍历数组的过程中保存两个值，一个是数组中的一个数字，另一个为次数。当比那里下一个数字的时候，若该数字与保存的数字一致，则将次数加一，否则减一，若此时次数小于0，则更新保存的数字，并将次数更改为0。最终保存的数字则为我们的候选数字，此时再遍历一遍数组进行验证。

```

```C++
class Solution {
public:
    int MoreThanHalfNum_Solution(vector<int> numbers) {
        if(numbers.size() == 0)
            return 0;
        int ans = numbers[0], cnt = 0;
        for(int i = 0; i < numbers.size(); ++ i){
            if(ans != numbers[i] && (--cnt) < 0){
                cnt = 0;
                ans = numbers[i];
            }
            if(ans == numbers[i])
                ++ cnt;
        }
        cnt = 0;
        for(int i = 0; i < numbers.size(); ++ i)
            if(numbers[i] == ans)
                ++ cnt;
        return (cnt > (numbers.size() >> 1) ? ans : 0);
    }
};


```


```python
class Solution:
    def MoreThanHalfNum_Solution(self, numbers):
        # write code here
        num, cnt = 0, 0
        for i in range(len(numbers)):
            if num == numbers[i]:
                cnt += 1
            else:
                cnt -= 1
                if cnt == -1:
                    cnt = 0
                    num = numbers[i]
        Numcnt = 0
        for i in range(len(numbers)):
            if numbers[i] == num:
                Numcnt += 1
        if Numcnt <= len(numbers) / 2:
            num = 0
        return num


```
