```
题目描述

在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置, 如果没有则返回 -1（需要区分大小写）.

解题思路

hash，遍历两遍字符串，第一遍记录各个字符出现的次数，第二遍检验字符是否只出现一次。
```

```C++
class Solution {
public:
    int FirstNotRepeatingChar(string str) {
        int visit[256] = {0};
        for(int i = 0; i < str.length(); ++ i)
            ++ visit[str[i]];
        for(int i = 0; i < str.length(); ++ i)
            if(visit[str[i]] == 1)
                return i;
        return -1;
    }
};


```

```python
class Solution:
    def FirstNotRepeatingChar(self, s):
        # write code here
        visit = [0] * 256
        for i in range(len(s)):
            visit[ord(s[i])] += 1
        for i in range(len(s)):
            if visit[ord(s[i])] == 1:
                return i
        return -1


```
