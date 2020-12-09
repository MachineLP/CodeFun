```
题目描述

求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。

解题思路

可以利用A&&B的隐含判断，当A错误，则不运行B
```

```C++
class Solution {
public:
    int Sum_Solution(int n) {
        int ans = n;
        ans && (ans += Sum_Solution(n-1));
        return ans;
    }
};


```


```python
class Solution:
    def Sum_Solution(self, n):
        # write code here
        ans = n
        temp = n and (self.Sum_Solution(n-1))
        ans += temp
        return ans


```
