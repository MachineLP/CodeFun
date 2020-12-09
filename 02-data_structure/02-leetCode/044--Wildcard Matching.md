
```
Given an input string (s) and a pattern (p), implement wildcard pattern matching with support for '?' and '*'.

'?' Matches any single character.
'*' Matches any sequence of characters (including the empty sequence).
The matching should cover the entire input string (not partial).

Note:

s could be empty and contains only lowercase letters a-z.
p could be empty and contains only lowercase letters a-z, and characters like ? or *.
Example 1:

Input:
s = "aa"
p = "a"
Output: false
Explanation: "a" does not match the entire string "aa".
Example 2:

Input:
s = "aa"
p = "*"
Output: true
Explanation: '*' matches any sequence.
Example 3:

Input:
s = "cb"
p = "?a"
Output: false
Explanation: '?' matches 'c', but the second letter is 'a', which does not match 'b'.
Example 4:

Input:
s = "adceb"
p = "*a*b"
Output: true
Explanation: The first '*' matches the empty sequence, while the second '*' matches the substring "dce".
Example 5:

Input:
s = "acdcb"
p = "a*c?b"
Output: false
```


```python
class Solution(object):
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        
        memo = {} 
        def dfs(l, r):
            if (l, r) in memo: return memo[l, r]
            if not r: return not l  
            if not l: return p[:r].count('*') == r 
            
            memo[l, r] = False
            if p[r-1] == '*':
                memo[l, r] = dfs(l-1, r) or dfs(l, r-1)    
            elif s[l-1] == p[r-1] or p[r-1] == '?':
                memo[l, r] = dfs(l-1, r-1)
            return memo[l, r] 
        return dfs(len(s), len(p))
        
        
        '''
        m, n = len(s), len(p)
        dp = [[False] * (n+1) for _ in range(m+1)]
        dp[0][0] = True       
        for j in range(1, n+1):
            if p[j-1] == '*' and dp[0][j-1]:
                dp[0][j] = True

        for i in range(1, m + 1): 
            for j in range(1, n + 1):
                if s[i-1] == p[j-1] or p[j-1] == '?':
                    dp[i][j] = dp[i-1][j-1]
                elif p[j-1] == '*':
                    dp[i][j] = dp[i][j-1] or dp[i-1][j]
        return dp[-1][-1]    
        '''
        
        '''
        if not p:
            return not s 
        if not s:
            return p.count('*') == len(p) 
        if s[-1] == p[-1] or p[-1] == '?':
            return self.isMatch(s[:-1], p[:-1])
        if p[-1] == '*':
            return self.isMatch(s[:-1], p) or self.isMatch(s, p[:-1]) 
        
        return False
        '''
    
        '''
        self.matched = False
        self.pattern = p
        self.plen = len(p)
        self.text = s
        self.tlen = len(s)
        
        self.rmatch(0, 0, self.text, self.tlen)
        
        return self.matched
    
    def rmatch(self, ti, pj, text, tlen):
        if self.matched: return
        if pj == self.plen:
            if ti == self.tlen:
                self.matched = True
            return
        if ti == self.tlen: 
            if pj == self.plen or self.pattern[pj] == '*':
                return self.rmatch(ti, pj+1, text, tlen)
            else:
                return False
        if self.pattern[pj] != '?' and self.pattern[pj] != '*' and self.pattern[pj] != self.text[ti]: return False
        
        if self.pattern[pj] == '*':
            for k in range(tlen-ti+1):
                if self.rmatch(ti+k, pj+1, text, tlen):
                    return True
            return False
        #elif self.pattern[pj] == '?':
        #    return self.rmatch(ti, pj+1, text, tlen)
        #    return self.rmatch(ti+1, pj+1, text, tlen)
        #elif ti < tlen and self.pattern[pj] == self.text[ti]:
        #    return self.rmatch(ti+1, pj+1, text, tlen)
        return self.rmatch(ti+1, pj+1, text, tlen)
    '''
            
'''
class Solution {
public:
    bool isMatch(string s, string p) {
        if(p.empty()) return s.empty();
        if(s.empty()) return p.empty() || p[0] == '*' ? isMatch(s, p.substr(1)) : false;
        if(p[0] != '?' && p[0] != '*' && p[0] != s[0]) return false; 
        if(p[0] == '*'){
            for(int i = 0; i <= s.size(); i++)
                if(isMatch(s.substr(i), p.substr(1))) return true;
            return false;
        }
        return isMatch(s.substr(1), p.substr(1));
    }
};

      class Solution {
public:
	bool isMatch(string s, string p) {
		if (s.size() == 0 && p.size() == 0) return true;
   		if (s.size() == 0)
		{
			bool flag = true;
			int n = p.size();
			while (n > 0)
			{
				flag &= (p[n - 1] == '*');
				n--;
			}
			return flag;
		}

		if (s.size() == 0 || p.size() == 0) return false;

		for (int i = 0; i < p.size(); i++)
		{
			if (p[i] == s[i])
			{
				return isMatch(s.substr(i + 1), p.substr(i + 1));
			}
			else if (p[i] == '?')
			{
				return isMatch(s.substr(i + 1), p.substr(i + 1));
			}
			else if (p[i] == '*')
			{
				bool flag = false;
				for (int j = 0; j <= s.size(); j++)
				{
					flag = isMatch(s.substr(j), p.substr(i + 1));
					if (flag == true) break;
				}
				return flag;
			}
			else
			{
				return false;
			}
		}

		return false;
	}
};

class Solution {
    // return value:
    // 0: reach the end of s but unmatched
    // 1: unmatched without reaching the end of s
    // 2: matched
    int dfs(string& s, string& p, int si, int pi) {
        if (si == s.size() and pi == p.size()) return 2;
        if (si == s.size() and p[pi] != '*') return 0;
        if (pi == p.size()) return 1;
        if (p[pi] == '*') {
            if (pi+1 < p.size() and p[pi+1] == '*') 
                return dfs(s, p, si, pi+1); // skip duplicate '*'
            for(int i = 0; i <= s.size()-si; ++i) {
                int ret = dfs(s, p, si+i, pi+1);
                if (ret == 0 or ret == 2) return ret; 
            }
        }
        if (p[pi] == '?' or s[si] == p[pi])
            return dfs(s, p, si+1, pi+1);
        return 1;
    }    
    
public:
    bool isMatch(string s, string p) {
        return dfs(s, p, 0, 0) > 1;
    }
};
'''
        
```

