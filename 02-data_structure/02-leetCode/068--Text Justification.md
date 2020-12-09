```
Given an array of words and a width maxWidth, format the text such that each line has exactly maxWidth characters and is fully (left and right) justified.

You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces ' ' when necessary so that each line has exactly maxWidth characters.

Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line do not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.

For the last line of text, it should be left justified and no extra space is inserted between words.

Note:

A word is defined as a character sequence consisting of non-space characters only.
Each word's length is guaranteed to be greater than 0 and not exceed maxWidth.
The input array words contains at least one word.
Example 1:

Input:
words = ["This", "is", "an", "example", "of", "text", "justification."]
maxWidth = 16
Output:
[
   "This    is    an",
   "example  of text",
   "justification.  "
]
Example 2:

Input:
words = ["What","must","be","acknowledgment","shall","be"]
maxWidth = 16
Output:
[
  "What   must   be",
  "acknowledgment  ",
  "shall be        "
]
Explanation: Note that the last line is "shall be    " instead of "shall     be",
             because the last line must be left-justified instead of fully-justified.
             Note that the second line is also left-justified becase it contains only one word.
Example 3:

Input:
words = ["Science","is","what","we","understand","well","enough","to","explain",
         "to","a","computer.","Art","is","everything","else","we","do"]
maxWidth = 20
Output:
[
  "Science  is  what we",
  "understand      well",
  "enough to explain to",
  "a  computer.  Art is",
  "everything  else  we",
  "do                  "
]
```

### 先分组，再排版。
```python
class Solution(object):
    def fullJustify(self, words, maxWidth):
        """
        :type words: List[str]
        :type maxWidth: int
        :rtype: List[str]
        """
        ans = []
        
        i = 0
        j = 0
        N = len(words)
        
        while i < N:
            j = i
            curr = 0
            col = []
            while j < N and len(words[j]) + curr <= maxWidth:
                curr += len(words[j])
                curr += 1 
                col.append(words[j])
                j += 1
                
            if j == N:
                ans.append(self.justify(col, maxWidth, False))
            else:
                ans.append(self.justify(col, maxWidth, True))
                
            i = j
            
        return ans
    
    def justify(self, col, maxWidth, full):
        wl = sum(map(len, col))
        wc = len(col)
            
        space = maxWidth - wl
        space_cnt = wc - 1
        
        if space_cnt == 0:
            return col[0] + (space * " ")
        
        base = space // space_cnt
        delta = space % space_cnt
        
        #print(space, space_cnt, base, delta)
                
        output = []
        
        if full:
            for i in range(0, len(col)):
                output.append(col[i])
                if i == len(col)-1:
                    continue

                if i < delta:
                    output.append(" " * (base+1))
                else:
                    output.append(" " * base)
        else:
            output = [" ".join(col), " " * (space - len(col) + 1)]
                
        return "".join(output)
```
