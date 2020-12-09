
```
Given a string, find the length of the longest substring without repeating characters.

Example 1:

Input: "abcabcbb"
Output: 3 
Explanation: The answer is "abc", with the length of 3. 
Example 2:

Input: "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.
Example 3:

Input: "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3. 
             Note that the answer must be a substring, "pwke" is a subsequence and not a substring.
```

```python
# Runtime: 800 ms, faster than 4.88% of Python online submissions for Longest Substring Without Repeating Characters.
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """

        d = {}
        longest=0
        cur = 0
        cur_length = 0
        while True:
            if cur == len(s):
                return max(cur_length, longest)
            elif s[cur] not in d:
                d[s[cur]] = cur
                cur = cur + 1
                cur_length = cur_length + 1
            else:
                if cur_length > longest:
                    longest = cur_length
                cur = d[s[cur]] + 1
                d = {}
                cur_length = 0

```

```python
# Runtime: 44 ms, faster than 99.58% of Python online submissions for Longest Substring Without Repeating Characters.
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        
        d = {}
        longest=0
        cur = 0
        cur_length = 0
        
        for index, i in enumerate(s):
            # d[i] >= cur 只关注上一次节点之后的。
            if i in d and d[i] >= cur:
                longest = max(cur_length, longest)
                cur_length = index - d[i]
                cur = d[i] + 1
            else:
                cur_length = cur_length + 1
            # 重复的会被覆盖哦
            d[i] = index
        
        return max(cur_length, longest)
                

```
