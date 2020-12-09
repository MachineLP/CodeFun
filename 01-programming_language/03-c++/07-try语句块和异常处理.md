
**try语句块和异常处理**

。。**异常**是指存在于运行时的反常行为，这些行为超出了函数正常功能的范围。典型的异常包括失去数据库连接以及遇到意外输入等。处理反常行为可能是设计所有系统最难的一部分。
。。当我们的某部分检测到一个它无法处理的问题时，需要用到异常处理。
。。在设计的时候，如果程序中含有可能引发的异常的代码，那么通常也会有专门的代码处理问题，例如：如果程序的问题时输入无效，则异常处理部分可能会要求用户重新输入正确的数据；如果丢失数据库的连接，会发出报警信息。
。。异常处理机制为程序中异常检测和异常处理这两部分的协作提供支持。
在c++中异常处理包括：
    ···**throw表达式**：异常检测部分使用throw表达式来表示它遇到了无法处理的问题，我们说throw引发了异常。
    ···**try语句块**：异常处理部分使用try语句块处理异常。try语句块以关键字try开始，并以一个或多个**catch子句** 结束，try语句块中代码抛出的异常通常会被某个句子处理。因为catch子句“处理”异常，所以它们也被称为**异常处理代码**。
    ···一套**异常类**：用于在throw表达式和相关的catch子句之间传递异常的具体信息。

例：
这个程序检查它读入的记录是否是关于同一个书籍的，如果不是引发一个异常；

```
Sales_item item1,item2;
// 首先检查item1和item2是否表示同一种书籍的
while(cin >> item1 >> item2)
{
    try{
         if(item.isbn() != item2.isbn())
         {
            //该异常类型是runtime_error异常。
            throw runtime_error("Data must refer to same ISBN");
         }
         // 如果执行到这里说明两个ISBN是相同的。
         cout << item1+item2 << endl;
    }
    catch(runtime_error err)
    {
       //提示用户两个ISBN必须一致，询问是否重新输入
       //err类型是runtime_error，因此能推断出what是runtime_error类的一个成员函数。
       cout << err.what() <<"\n Try Again? Enter y or n" << endl;
       char c;
       cin >> c;
       if(!cin || c == 'n')
          break;  // 跳出while循环。
    }
}
```

异常处理注意的其他问题在c++primer第五版175页。
