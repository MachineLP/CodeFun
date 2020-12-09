
```
Given two strings A and B of lowercase letters, return true if and only if we can swap two letters in A so that the result equals B.
 

Example 1:

Input: A = "ab", B = "ba"
Output: true
Example 2:

Input: A = "ab", B = "ab"
Output: false
Example 3:

Input: A = "aa", B = "aa"
Output: true
Example 4:

Input: A = "aaaaaaabc", B = "aaaaaaacb"
Output: true
Example 5:

Input: A = "", B = "aa"
Output: false
 

Note:

0 <= A.length <= 20000
0 <= B.length <= 20000
A and B consist only of lowercase letters.
```

## Solution
```python
class Solution:
    def buddyStrings(self, A, B):
        if len(A) == len(B):
            pair = None
            swaps = False
            uniq = set()
            for a, b in zip(A, B):
                if a == b:
                    uniq.add(a)
                elif pair is None:
                    pair = a, b
                elif swaps:
                    return False
                else:
                    x, y = pair
                    if x == b and a == y:
                        swaps = True
                    else:
                        return False
            return swaps or len(uniq) < len(A)
        else:
            return False
```
