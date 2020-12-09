
```
The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)

P   A   H   N
A P L S I I G
Y   I   R
And then read line by line: "PAHNAPLSIIGYIR"

Write the code that will take a string and make this conversion given a number of rows:

string convert(string s, int numRows);
Example 1:

Input: s = "PAYPALISHIRING", numRows = 3
Output: "PAHNAPLSIIGYIR"
Example 2:

Input: s = "PAYPALISHIRING", numRows = 4
Output: "PINALSIGYAHRPI"
Explanation:

P     I    N
A   L S  I G
Y A   H R
P     I
```


```python
# 
class Solution(object):
    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        length = len(s)
        result = ""
        hashTable = [""] * numRows

        if numRows >= length or numRows == 1:
            return s

        for i in range(length):
            index = i%(2*numRows-2)
            if index < numRows:
                hashTable[index] += s[i]
            else:
                hashTable[2*numRows-2-index] += s[i]

        for i in range(numRows):
            result += hashTable[i]

        return result
```
