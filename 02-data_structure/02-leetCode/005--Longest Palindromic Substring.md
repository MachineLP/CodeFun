
```
Given a string s, find the longest palindromic substring in s. You may assume that the maximum length of s is 1000.

Example 1:

Input: "babad"
Output: "bab"
Note: "aba" is also a valid answer.
Example 2:

Input: "cbbd"
Output: "bb"
```


```python
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        res = ""
        for i in xrange(len(s)):
            # odd case, like "aba"
            tmp = self.helper(s, i, i)
            if len(tmp) > len(res):
                res = tmp
            # even case, like "abba"
            tmp = self.helper(s, i, i+1)
            if len(tmp) > len(res):
                res = tmp
        return res
     
    # get the longest palindrome, l, r are the middle indexes   
    # from inner to outer
    def helper(self, s, l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1; r += 1
        return s[l+1:r]
```


```python
# 还可以再优化，可这里：https://mp.weixin.qq.com/s/t7Q0slX3q8Qlhg8F8pXrZQ 
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
       
        # 预处理字符串， 加 '#'
        ss = self.pre_handle_string(s)
        # 处理后的字符长度
        ss_len = len(ss)
        # 右边界
        right_side = 0
        # 右边界对应的回文串中心
        right_side_center = 0
        # 保存以每个字符为中心的回文长度一半(向下取正)
        half_len_arr = [0 for i in range(ss_len)]
        # 记录回文中心
        center = 0
        # 记录最长回文长度
        longest_half = 0
    
        for i in range (ss_len):
            # 是否需要中心扩展
            need_calc = True
            # 如果在右边界的覆盖之内
            if right_side > i:
                # 计算相对right_side_center的对称位置
                left_center = 2*right_side_center - i
                # 根据回文性质得出结论
                half_len_arr[i] = half_len_arr[left_center]
                # 如果超过了右边界，进行调整
                if i+half_len_arr[i] > right_side:
                    half_len_arr[i] = right_side-i
                # 如果根据已知条件计算得出的最长回文小于右边界，则不需要扩展了
                if i+half_len_arr[left_center] < right_side:
                    need_calc = False
            # 中心扩展
            if need_calc:
                while (i-1-half_len_arr[i]>=0) and (i+1+half_len_arr[i] < ss_len):
                    if ss[i+1+half_len_arr[i]] == ss[i-1-half_len_arr[i]]:
                        half_len_arr[i] = half_len_arr[i]+1
                    else:
                        break
                
                # 更新右边界及中心
                right_side = i + half_len_arr[i]
                right_side_center = i
                # 记录最长回文串
                if half_len_arr[i] > longest_half:
                    center = i
                    longest_half = half_len_arr[i]
        sb = ''
        # +2是因为去掉之间添加的#
        for i in range (center-longest_half+1,center+longest_half,2):
            sb = sb + ss[i]
    
        return str(sb)
    def pre_handle_string(self, s):
        sb = '#'
        s_len = len(s)
        for i in range(s_len):
            sb = sb + s[i] + '#'
        return str(sb)
```
