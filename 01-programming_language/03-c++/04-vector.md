
**标准库类型vector**

使用vector, 必须包含头文件`#include <vector>` `using std::vector;`

c++语言既有类模板，也有函数模板，其中**vector**就是一个类模板。
       模板本身不是类或函数，相反可以将模板看作为编译器生成类或者函数编写的一份说明。编译器通过模板创建类或者函数的过程称为**实例化，当使用模板时，需要指出编译器应把类或函数实例化成何种类型**。
       对于类模板来说，我们通过提供一些额外信息来指定模板到底实例化成什么样的类，需要提供哪些信息由模板来定。提供信息的方式总是这样的：**即在模板名字后面跟一对尖括号，在括号内放上信息。**
      

```
 例：
       vector<int> ivec;  // ivec保存int类型的对象
       vector<Sales_item> Sales_vec;  // 保存Sales_item类型的对象。
       vector<vector<string>> file;  //该向量的元素是vector对象
```
**定义和初始化vector对象**
![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTUwNzE0MjI0MzU5MjY1?x-oss-process=image/format,png)

**其他vector操作**
![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTUwNzE1MDkyMTMxMTIy?x-oss-process=image/format,png)


**注：**如果循环体内部包含有向vector对象添加的语句，则不能使用范围for循环。

```
例：
vector<int> v{1,2,3,4,5,6,7};
for(auto i : v)
     v.push_back(i+1);  //错误，范围for循环体内不应改变其所遍历序列的大小。
```
**不能使用下标形式添加元素**

```
例：
vector<int> ivec;  // 空 vector对象
for(decltype(ivec.size()) ix=0; ix!=10; ++ix）
    ivec[ix] = ix;  // 严重错误：ivec不包含任何元素
     // 正确的方法应该使用ivec.push_back(ix);
```
vector对象（以及string对象）的下标运算符可用于访问已存在的元素，而不能用于添加元素。