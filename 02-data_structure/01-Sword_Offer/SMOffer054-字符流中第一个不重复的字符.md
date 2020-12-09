```
题目描述

请实现一个函数用来找出字符流中第一个只出现一次的字符。例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"。当从该字符流中读出前六个字符“google"时，第一个只出现一次的字符是"l"。

输出描述:

如果当前字符流没有存在出现一次的字符，返回#字符。
解题思路

hash，第一遍历的时候记录每个字符出现次数，第二次遍历的时候遇到第一个出现次数为1的字符即返回。

```


```C++
class Solution
{
public:
  //Insert one char from stringstream
    void Insert(char ch)
    {
         s += ch;
         ++ visit[ch];
    }
  //return the first appearence once char in current stringstream
    char FirstAppearingOnce()
    {
        for(int i = 0; i < s.length(); ++ i)
            if(visit[s[i]] == 1)
                return s[i];
        return '#';
    }
private:
    string s;
    int visit[256] = {0};
};


```


```python
class Solution:
    # 返回对应char
    def __init__(self):
        self.s = ''
        self.visit = [0] * 256
    def FirstAppearingOnce(self):
        # write code here
        Len = len(self.s)
        for i in range(Len):
            if self.visit[ord(self.s[i])] == 1:
                return self.s[i]
        return '#'
    def Insert(self, char):
        # write code here
        self.s += char
        self.visit[ord(char)] += 1


```
