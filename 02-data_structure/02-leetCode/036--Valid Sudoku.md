
```
Determine if a 9x9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:

Each row must contain the digits 1-9 without repetition.
Each column must contain the digits 1-9 without repetition.
Each of the 9 3x3 sub-boxes of the grid must contain the digits 1-9 without repetition.

A partially filled sudoku which is valid.

The Sudoku board could be partially filled, where empty cells are filled with the character '.'.

Example 1:

Input:
[
  ["5","3",".",".","7",".",".",".","."],
  ["6",".",".","1","9","5",".",".","."],
  [".","9","8",".",".",".",".","6","."],
  ["8",".",".",".","6",".",".",".","3"],
  ["4",".",".","8",".","3",".",".","1"],
  ["7",".",".",".","2",".",".",".","6"],
  [".","6",".",".",".",".","2","8","."],
  [".",".",".","4","1","9",".",".","5"],
  [".",".",".",".","8",".",".","7","9"]
]
Output: true
Example 2:

Input:
[
  ["8","3",".",".","7",".",".",".","."],
  ["6",".",".","1","9","5",".",".","."],
  [".","9","8",".",".",".",".","6","."],
  ["8",".",".",".","6",".",".",".","3"],
  ["4",".",".","8",".","3",".",".","1"],
  ["7",".",".",".","2",".",".",".","6"],
  [".","6",".",".",".",".","2","8","."],
  [".",".",".","4","1","9",".",".","5"],
  [".",".",".",".","8",".",".","7","9"]
]
Output: false
Explanation: Same as Example 1, except with the 5 in the top left corner being 
    modified to 8. Since there are two 8's in the top left 3x3 sub-box, it is invalid.
Note:

A Sudoku board (partially filled) could be valid but is not necessarily solvable.
Only the filled cells need to be validated according to the mentioned rules.
The given board contain only digits 1-9 and the character '.'.
The given board size is always 9x9.
```


```python
class Solution(object):
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        import numpy as np
        cnt = 9
        row_mask = np.zeros([cnt,cnt]) #cnt * [cnt * [False]]
        col_mask = np.zeros([cnt,cnt]) #cnt * [cnt * [False]]
        area_mask = np.zeros([cnt,cnt]) #cnt * [cnt * [False]]
        # check each rows and cols
        for r in range ( len(board) ):
            for c in range ( len(board[r]) ):
                if board[r][c].isdigit() == False:
                    continue
                idx = int( board[r][c] ) - 1
                
                # check the rows
                if row_mask[r][idx] == 1:
                    return False
                row_mask[r][idx] = 1
                
                # check the cols
                if col_mask[c][idx] == 1:
                    return False
                col_mask[c][idx] = 1
                
                #check the area
                area = r/3 * 3 + c/3
                if area_mask[area][idx] == 1:
                    return False
                area_mask[area][idx] = 1
        return True
        
```
