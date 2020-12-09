```
题目描述

将一个字符串转换成一个整数(实现Integer.valueOf(string)的功能，但是string不符合数字要求时返回0)，要求不能使用字符串转换整数的库函数。 数值为0或者字符串不是一个合法的数值则返回0。

输入描述:

输入一个字符串,包括数字字母符号,可以为空
输出描述:

如果是合法的数值表达则返回该数字，否则返回0
输入

+2147483647
    1a33
输出

2147483647
    0
解题思路

题目本身不难，要考虑几种情况，首先不合要求的字符串，其次负数字符串。

```


```C++
class Solution {
public:
    int StrToInt(string str) {
        int ans = 0, i = 0, flag = 0;
        if(str[0] == '-' || str[0] == '+'){
            ++ i;
            flag = str[0] == '-' ? 1 : 0;
        }
        for(; i < str.length(); ++ i){
            if(!isdigit(str[i]))
                return 0;
            if(flag == 1)
                ans = ans * 10 - (str[i] - '0');
            else
                ans = ans * 10 + (str[i] - '0');
        }
        return ans;
    }
};


```


```python
class Solution:
    def StrToInt(self, s):
        # write code here
        if len(s) == 0:
            return 0
        ans, cur, flag = 0, 0, 1
        if s[0] == '-' or s[0] == '+':
            if s[0] == '-':
                flag = -1
            cur += 1
        while cur < len(s):
            if s[cur] < '0' or s[cur] > '9':
                return 0
            ans = ans * 10 + flag * (ord(s[cur]) - ord('0'))
            cur += 1
        return ans


```
