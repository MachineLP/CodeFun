
```
You are given a string, s, and a list of words, words, that are all of the same length. Find all starting indices of substring(s) in s that is a concatenation of each word in words exactly once and without any intervening characters.

Example 1:

Input:
  s = "barfoothefoobarman",
  words = ["foo","bar"]
Output: [0,9]
Explanation: Substrings starting at index 0 and 9 are "barfoor" and "foobar" respectively.
The output order does not matter, returning [9,0] is fine too.
Example 2:

Input:
  s = "wordgoodgoodgoodbestword",
  words = ["word","good","best","word"]
Output: []
```


```python

class Solution(object):
    def findSubstring(self, s, words):
        """
        :type s: str
        :type words: List[str]
        :rtype: List[int]
        """
        res = []
        if (len(s) <=0 or len(words) <=0 ):
            return res
        
        n = len(s)
        m = len(words)
        l = len(words[0])
        
        # put all the words into a map
        expected = {}
        for i in range(m):
            if words[i] in expected:
                expected[words[i]] += 1
            else:
                expected[words[i]] = 1
        
        for i in range(l):
            actual = {}
            count = 0
            win_left = i
            for j in range(i, n-l+1, l):
                # substr
                word = s[j:j+l]
               
                # if not found, then restart from j+1
                # 说明子串 不在模式串中， 进行初始化
                if word not in expected:
                    actual = {}
                    count = 0
                    win_left = j+l
                    continue
                # 说明子串 在模式串中有， 对应加1
                count += 1
                # 统计word字的数量
                if word not in actual:
                    actual[word] = 1
                else:
                    actual[word] += 1
                
                # If there is more appearance of 'word' than expected
                # 因为这里是大于模式串的，count -= 1，actual[tmp] -= 1，win_left += l
                if actual[word]>expected[word]:
                    tmp = ''
                    while tmp!=word:
                        tmp = s[win_left:win_left+l]
                        count -= 1
                        actual[tmp] -= 1
                        win_left += l

                
                if count == m:
                    res.append(win_left)
                    tmp = s[win_left:win_left+l]
                    actual[tmp] -= 1
                    win_left += l
                    count -= 1
        return res
        
        
        '''
        # the solution should be appended in the below list
        result = []
        
        # ignoring any action if the string or the list of words has 0 length
        if len(s) == 0 or len(words) == 0:
            return result
        
        from collections import Counter
        
        word_length = len(words[0])
        words_count = len(words)
        
        # the window size that we are going to slide
        window_length = words_count * word_length
   
        # temp list to store splitted window size part of the string 
        temp = []
   
        # checking from 0 till word length before the sting end   
        for i in range(0,len(s) - window_length + 1):
            # the sliding window from the string to be checked against the words
            word_window = s[i:i + window_length]
            # splitting the window string to a list of splitted words
            temp = [word_window[j:j+word_length] for j in range(0, window_length, word_length)]
            # checking if the splitted words and original words are the same
            if Counter(temp) == Counter(words):
                # append the value of i which is the index of the first letter in the sliding window
                result.append(i)
    
        return result
        '''
```
