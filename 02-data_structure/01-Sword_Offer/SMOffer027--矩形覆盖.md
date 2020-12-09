```
题目描述

输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。

输入描述:

输入一个字符串,长度不超过9(可能有字符重复),字符只包括大小写字母。
解题思路

把字符串分为两个部分一部分为字符串的第一个字符，另一部分为第一个字符后所有的字符，递归求第一个字符后所有的字符可能的排列，然后将第一个字符与后面的字符逐一交换，求得每个字符作为首位可能有的排列。这里要注意一点，就是字符串可能有重复字符，所以在交换的时候，加上一个判断，避免重复劳动。

```

```C++
class Solution {
public:
    vector<string>ans;
    vector<string> Permutation(string str) {
        ans.clear();
        if(str.length() > 0)
            MyPermutation(str, 0);
        return ans;
    }
    void MyPermutation(string str, int cur){
        if(cur == str.length()){
            ans.push_back(str);
            return;
        }
        for(int i = cur; i < str.length(); ++ i){
            if(i != cur && str[i] == str[cur])
                continue;
            swap(str[i], str[cur]);
            MyPermutation(str, cur+1);
        }
    }
};


```

```python
class Solution:
    def __init__(self):
        self.ans = []
    def Permutation(self, ss):
        # write code here
        if len(ss) > 0:
            self.myPermutation(ss, 0)
        return self.ans
    def myPermutation(self, s, cur):
        if cur == len(s):
            self.ans.append(s)
            return
        for i in range(cur, len(s)):
            if i != cur and s[i] == s[cur]:
                continue
            s_list = list(s)
            s_list[i], s_list[cur] = s_list[cur], s_list[i]
            s = ''.join(s_list)
            self.myPermutation(s, cur+1)


```
