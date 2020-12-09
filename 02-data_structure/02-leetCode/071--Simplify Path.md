```
Given an absolute path for a file (Unix-style), simplify it. Or in other words, convert it to the canonical path.

In a UNIX-style file system, a period . refers to the current directory. Furthermore, a double period .. moves the directory up a level. For more information, see: Absolute path vs relative path in Linux/Unix

Note that the returned canonical path must always begin with a slash /, and there must be only a single slash / between two directory names. The last directory name (if it exists) must not end with a trailing /. Also, the canonical path must be the shortest string representing the absolute path.

 

Example 1:

Input: "/home/"
Output: "/home"
Explanation: Note that there is no trailing slash after the last directory name.
Example 2:

Input: "/../"
Output: "/"
Explanation: Going one level up from the root directory is a no-op, as the root level is the highest level you can go.
Example 3:

Input: "/home//foo/"
Output: "/home/foo"
Explanation: In the canonical path, multiple consecutive slashes are replaced by a single one.
Example 4:

Input: "/a/./b/../../c/"
Output: "/c"
Example 5:

Input: "/a/../../b/../c//.//"
Output: "/c"
Example 6:

Input: "/a//b////c/d//././/.."
Output: "/a/b/c"
```


## 就这个代码手撸了一个小时，真是让人蛋疼啊
```python
class Solution(object):
    def simplifyPath(self, path):
        """
        :type path: str
        :rtype: str
        """
        # path = '/a/./b/../../c/'
        if path == '':
            return
        path_str = ''
        path_list = []
        path_lengh = len(path)
        i = 0
        print ('111', path )
        while i < path_lengh:
            print ('222', i )
            flag = 0
            while i < path_lengh and path[i] == '/':
                i = i+1
                flag = 1
            while i < path_lengh and path[i] != '/':
                path_str = path_str + path[i]
                i = i+1
            if flag == 1 and path_str!='':
                path_list.append( path_str )
                path_str = ''
        print ('>>>', path_list)
        
        res_list = []
        for index, per in enumerate(path_list):
            if per == '.':
                continue
            elif per == '..':
                try:
                    res_list.pop(-1)
                except:
                    continue
            else:
                res_list.append(per)
        print ('res_list:', res_list)
        
        return '/' + '/'.join(res_list)
            
                
```

