
 ![image](https://assets.leetcode.com/uploads/2018/10/12/8-queens.png)

```
The n-queens puzzle is the problem of placing n queens on an n×n chessboard such that no two queens attack each other.

Given an integer n, return all distinct solutions to the n-queens puzzle.

Each solution contains a distinct board configuration of the n-queens' placement, where 'Q' and '.' both indicate a queen and an empty space respectively.

Example:

Input: 4
Output: [
 [".Q..",  // Solution 1
  "...Q",
  "Q...",
  "..Q."],

 ["..Q.",  // Solution 2
  "Q...",
  "...Q",
  ".Q.."]
]
Explanation: There exist two distinct solutions to the 4-queens puzzle as shown above.
```


```python
class Solution(object):
    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        res = []
        self.dfs([-1]*n, 0, [], res)
        return res
 
    # nums is a one-dimension array, like [1, 3, 0, 2] means
    # first queen is placed in column 1, second queen is placed
    # in column 3, etc.
    def dfs(self, nums, index, path, res):
        if index == len(nums):
            res.append(path)
            return  # backtracking
        for i in xrange(len(nums)):
            # 第index行第i列； index表示的是行，nums中的数值表示的是列。 
            nums[index] = i
            if self.valid(nums, index):  # pruning
                tmp = "."*len(nums)
                self.dfs(nums, index+1, path+[tmp[:i]+"Q"+tmp[i+1:]], res)

    # check whether nth queen can be placed in that column
    def valid(self, nums, n):
        for i in xrange(n):
            if abs(nums[i]-nums[n]) == n -i or nums[i] == nums[n]:
                return False
        return True
    
    '''
        self.BOARD_SIZE = n
        self.solution_count = 0
        self.queen_list = [0] * self.BOARD_SIZE
        self.eight_queens(0)
        return self.queen_list
    
    def eight_queens(self, cur_column):
        
        if cur_column >= self.BOARD_SIZE:
            self.solution_count += 1
            # 解
            # return self.queen_list
            return 
        else:
            for i in range(self.BOARD_SIZE):
                if self.is_valid_pos(cur_column, i):
                    self.queen_list[cur_column] = i
                    self.solveNQueens(cur_column + 1)

    def is_valid_pos(self, cur_column, pos):
        """
        因为采取的是每列放置1个皇后的做法
        所以检查的时候不必检查列的合法性，只需要检查行和对角
        1. 行：检查数组在下标为cur_column之前的元素是否已存在pos
        2. 对角：检查数组在下标为cur_column之前的元素，其行的间距pos - QUEEN_LIST[i]
           和列的间距cur_column - i是否一致
        :param cur_column:
        :param pos:
        :return:
        """
        i = 0
        while i < cur_column:
            # 同行
            if self.queen_list[i] == pos:
                return False
            # 对角线
            if cur_column - i == abs(pos - self.queen_list[i]):
                return False
            i += 1
        return True
    '''
```
