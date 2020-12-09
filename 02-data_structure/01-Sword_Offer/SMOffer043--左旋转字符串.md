```
题目描述

汇编语言中有一种移位指令叫做循环左移（ROL），现在有个简单的任务，就是用字符串模拟这个指令的运算结果。对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。是不是很简单？OK，搞定它！

解题思路

以abcXYZdef为例，我们先分别翻转前3个字符以及后面的字符，得到cbafedZYX，再将整个字符串进行翻转，得到XYZdefabc。

```


```C++
class Solution {
public:
    string LeftRotateString(string str, int n) {
        int len = str.length();
        if(len == 0 || n > len)
            return str;
        reverse(str.begin(), str.begin() + n);
        reverse(str.begin() + n, str.end());
        reverse(str.begin(), str.end());
        return str;
    }
};


class Solution {
public:
    string LeftRotateString(string str, int n) {
        int len = str.length();
        if(len == 0 || n > len)
            return str;
        string ans = str.substr(n);
        ans += str.substr(0, n);
        return ans;
    }
};


```

```python
class Solution:
    def LeftRotateString(self, s, n):
        # write code here
        if n == 0 or len(s) < n:
            return s
        temps = s[n-1::-1] + s[-1:n-1:-1]
        return ''.join(reversed(temps))


```
