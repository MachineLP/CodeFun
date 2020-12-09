```
题目描述

请实现一个函数用来匹配包括'.'和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（包含0次）。 在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配

解题思路

当模式中第二个字符不是*，若字符串对应的字符与模式中对应的字符匹配（相同或者模式中为.），字符串和模式各向后移动一个字符，然后匹配剩余的字符串和模式。否则直接返回False

当模式中第二个字符为*，①模式直接向后移动两个字符，②字符串对应的字符与模式中对应的字符匹配，字符向后移动一个，模式可以保持不动或者向后移动两个

```

```C++
class Solution {
public:
    bool match(char* str, char* pattern)
    {
        if(str == nullptr || pattern == nullptr)
            return false;
        return matchCore(str,pattern);
    }
    bool matchCore(char* str, char* pattern){
        if(*str == '\0' && *pattern == '\0')
            return true;
        if(*str != '\0' && *pattern == '\0')
            return false;
        if(*(pattern + 1) == '*'){
            if(*pattern == *str || (*str != '\0' && *pattern == '.'))
                return matchCore(str+1, pattern+2) || matchCore(str+1, pattern) || matchCore(str, pattern+2);
            else
                return matchCore(str, pattern+2);
        }
        if(*str == *pattern || (*pattern == '.' && *str != '\0'))
            return matchCore(str+1, pattern+1);
        return false;
    }
};


```

```python
class Solution:
    # s, pattern都是字符串
    def match(self, s, pattern):
        # write code here
        return self.matchCore(s, pattern, 0, 0)
    def matchCore(self, s, pattern, i, j):
        if i == len(s) and j == len(pattern):
            return True
        if j >= len(pattern) or i > len(s):
            return False
        if j < len(pattern)-1 and pattern[j+1] == '*':
            if i == len(s) or pattern[j] == s[i] or pattern[j] == '.':
                return self.matchCore(s,pattern,i,j+2) or self.matchCore(s,pattern,i+1,j) or self.matchCore(s,pattern,i+1,j+2)
            else:
                return self.matchCore(s,pattern,i,j+2)
        if i == len(s) or pattern[j] == s[i] or pattern[j] == '.':
            return self.matchCore(s,pattern,i+1,j+1)
        return False


``
