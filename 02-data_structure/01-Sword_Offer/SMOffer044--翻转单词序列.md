```
题目描述

牛客最近来了一个新员工Fish，每天早晨总是会拿着一本英文杂志，写些句子在本子上。同事Cat对Fish写的内容颇感兴趣，有一天他向Fish借来翻看，但却读不懂它的意思。例如，“student. a am I”。后来才意识到，这家伙原来把句子单词的顺序翻转了，正确的句子应该是“I am a student.”。Cat对一一的翻转这些单词顺序可不在行，你能帮助他么？

解题思路

首先翻转整个句子，其次再翻转每个单词中的字符的顺序

```

```C++
class Solution {
public:
    string ReverseSentence(string str) {
        reverse(str.begin(), str.end());
        int lastpos = 0;
        for(int i = 1; i < str.length(); ++ i){
            while(i < str.length() && str[i] != ' ')
                ++ i;
            reverse(str.begin() + lastpos, str.begin() + i);
            lastpos = i + 1;
        }
        return str;
    }
};


class Solution {
public:
    string ReverseSentence(string str) {
        string ans = "", temp;
        istringstream is(str);
        while(is >> temp)
            ans = temp + ' ' + ans;
        if(ans.length() > 0)
            ans.pop_back();
        if(ans.length() == 0 && str.length() > 0)
            return str;
        return ans;
    }
};


```

```python
class Solution:
    def ReverseSentence(self, s):
        # write code here
        s, ans = s[::-1], ''
        s = s.split(' ')
        for i in range(len(s)):
            ans += s[i][::-1]
            if i != len(s) - 1:
                ans += ' '
        return ans


```
