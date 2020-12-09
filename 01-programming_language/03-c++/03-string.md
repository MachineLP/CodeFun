
**标准库类型string**

标准库类型**string**表示可边长字符序列，使用**string**类型必须**包含string头文件**。**string**定义在命名空间**std**中。

```
 #include <string>
 using namespace std;
```
**定义和初始化string对象**

      **初始化string对象的方式**
```
      string s1;  // 默认初始化，s1是一个空串
      string s2(s1);  //s2是s1的副本
      string s2 = s1;  // 等价于s2(s1),s2是s1的一个副本
      string s3("value");  // s3是字面值"value"的副本，除了字面值最后的那个空字符串外
      string s3 = "value";  // 等价于s3("value")
      string s4(n,'c');  // 把s4初始化为由连续n个字符c组成的串
      string s5 = string(n,'c');
```
如果使用**等号（=）**初始化一个变量，实际上执行的是**拷贝初始化**。
如果不使用等号，则执行的是**直接初始化**。

**string对象上的操作**

```
string s;  
os << s;  // 将s写到输出流os中，返回os
is >> s;  // 从is中读取字符串赋给s，字符串以空白分割，返回is
getline(is,s);  // 从is中读取一行赋给s，返回is
s.empty();  // s为空返回true，否则返回false。
s.size();  // 返回s中的字符个数。
s[n];  // 返回s中第n个字符的引用，位置n从0记起
s1 + s2;  //  返回s1和s2连接后的结果
s1 = s2;  // 用s2的副本代替s1中原本的字符
s1 == s2;  // 如果s1和s2中所含的字符完全一样,则它们相等
s1 != s2;   // string字符的相等性判断对字母的大小写敏感
< , <= , > , >=;  // 利用字符在字典中的顺序进行比较，且字母的大小写敏感
```
**读取未知数量的string对象**

```
int main()
{
    string word；
    while(cin>>word)     
       cout << word <<endl;
    return 0 ;
}
```
**使用getline读取一整行**
getline只要一遇到转行符就结束并返回结果，哪怕输入的一开始就是换行符（这将得到一个空的string）。

```
int main()
{
    string line;
    // 每次读入一整行，直至到达文件末尾。
    while(getline(cin，line))     
       cout << line <<endl;
    return 0 ;
}
```
**string::size_type**
size函数返回类型是一个**string::size_type**类型的值，是无符号的整型，**因此切忌，如果表达式中混用了带符号和无符号将可能产生意想不到的结果。

```
例如：假设n是一个int负值，则表达式s.size()<n的判断几乎肯定是true。
// 如果一个表达式有了size()函数就不要用int了，这样可以避免混用int和unsigned可能带来的问题。

// 具体使用的时候，通过作用域操作符来表明**名字size_type是类string中定义**。
// 在不知道size函数返回类型返回值情况下，C++11新标准：
auto len = line.size(); // len类型是string::size_type
```
**string相加**

```
string s1 = "hello,", s2="world\n";
string s3 = s1 + s2;  // s3的内容是hello,world\n
string s1+=s2;  // s1 = s1+s2;
string s1 += "world\n";
string s4 = "hello" + "," +s2; //错误：不能把字面值直接相加
 // 相加时，必须保证每个加法运算符（+）的两侧的运算对象至少有一个是string。
```



**处理每个字符？ 使用基于范围的for语句**
目前最好的办法是使用C++11新标准提供的语句：范围for（range for）语句。
其语法：

```
for(declaration : expression)
     statement
```
其中，expression部分是一个对象，用于表示一个序列。declaration部分负责定义一个变量，该变量将被用于访问序列的基础元素。每次迭代，declaration部分的变量会被初始化为expression部分的下一个元素。

```
例子：
string str("some string");
// 每行输出str中的一个字符。
for(auto c : str)   // 对于str中的每个字符
     cout << c << endl;   // 输出当前字符，后面紧跟一个换行符
```
**使用范围for语句改变字符串中的字符**
如果想要改变string对象中字符的值，必须把循环变量定义成引用类型。

```
例子：
string s("Hello world!!!");
// 转换成大写形式。
for(auto &c : s)  //对于s中的每个字符（注意：c是引用）
      c = toupper(c);  // c是一个引用，因此赋值语句将改变s中的字符的值
cout << s << endl;
```
**只处理一部分字符**
想访问string对象中的**单个字符有两种形式**：一种是下标，另一种是使用迭代器。
**string对象的下标必须大于等于0而小于s.size()。**

```
// 使用下标：
string s("some string");
if(!s.empty())
    s[0] = toupper(s[0]);  // 输出为Some string
```

```
//使用迭代
for (decltype(s.size()) index = 0; index != s.size() && !isspace(s[index]); 
++index)
     s[index] = toupper(s[index]);  //输出为SOME string