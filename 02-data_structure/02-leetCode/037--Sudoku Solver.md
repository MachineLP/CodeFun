
```
Write a program to solve a Sudoku puzzle by filling the empty cells.

A sudoku solution must satisfy all of the following rules:

Each of the digits 1-9 must occur exactly once in each row.
Each of the digits 1-9 must occur exactly once in each column.
Each of the the digits 1-9 must occur exactly once in each of the 9 3x3 sub-boxes of the grid.
Empty cells are indicated by the character '.'.


A sudoku puzzle...


...and its solution numbers marked in red.

Note:

The given board contain only digits 1-9 and the character '.'.
You may assume that the given Sudoku puzzle will have a single unique solution.
The given board size is always 9x9.
```


```python
import numpy as np
cnt = 9
row_mask = np.zeros([cnt,cnt]) #cnt * [cnt * [False]]
col_mask = np.zeros([cnt,cnt]) #cnt * [cnt * [False]]
area_mask = np.zeros([cnt,cnt]) #cnt * [cnt * [False]]

class Solution(object):
    
    def solveSudoku(self, board):
        rows, cols, triples, visit = collections.defaultdict(set), collections.defaultdict(set), collections.defaultdict(set), collections.deque([])
        for r in range(9):
            for c in range(9):
                if board[r][c] != ".":
                    rows[r].add(board[r][c])
                    cols[c].add(board[r][c])
                    triples[(r // 3, c // 3)].add(board[r][c])
                else:
                    visit.append((r, c))
        def dfs():
            if not visit:
                return True
            r, c = visit[0]
            t = (r // 3, c // 3)
            for dig in {"1", "2", "3", "4", "5", "6", "7", "8", "9"}:
                if dig not in rows[r] and dig not in cols[c] and dig not in triples[t]:
                    board[r][c] = dig
                    rows[r].add(dig)
                    cols[c].add(dig)
                    triples[t].add(dig)
                    visit.popleft()
                    if dfs():
                        return True
                    else:
                        board[r][c] = "."
                        rows[r].discard(dig)
                        cols[c].discard(dig)
                        triples[t].discard(dig)
                        visit.appendleft((r, c))
            return False
        dfs()
    
    '''
    
    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        if ( self.initSudokuMask(board) == False ):
            return
        self.recursiveSudoKu(board, 0, 0)
    
    def recursiveSudoKu(self, board, row, col):
        if row >= cnt:
            return True
        if col >= cnt:
            return self.recursiveSudoKu(board, row+1, 0)
        
        if board[row][col] != '.':
            return self.recursiveSudoKu(board, row, col+1)
        
        # pick a number for empty cell
        for i in range(cnt):
            area = row/3 * 3 + col/3
            if row_mask[row][i]==1 or col_mask[col][i]==1 or area_mask[area][i]==1:
                continue
            # set the number and sovle it recursively
            board[row][col] = str ( i + 1 )
            row_mask[row][i] = col_mask[col][i] = area_mask[area][i] = 1
            if self.recursiveSudoKu(board, row, col+1) == True:
                return True
            # backtrace
            board[row][col] = '.'
            row_mask[row][i] = col_mask[col][i] = area_mask[area][i] = 0
        return False
            
    
    def initSudokuMask(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        
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
        '''
```

