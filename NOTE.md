## 2023/10/28

### 测试网络请求
    - 注解学习(放到Mapping注解的方法参数列表中)
        - @RequestParam("name") String name
            - 用来接收前端form表单的数据
        - @ModelAttribute User user
            - 可以实现form表单的数据映射
        - @RequestBody User user
            - 实现Post请求的数据映射**（不能是form表单，一个大坑，早上埋了好久）**

### 测试文件上传
    - 相关类
        - MultipartFile
            - 解释
                - 是springboot用来处理文件上传的一个类
            - 相关API
                - getOriginalFilename() -- 获取原始文件的文件名
                - getBytes() -- 将文件转换成字节数组
                - getInputStream() -- 将文件转换成字节输入流
                - getContentType() -- 获取上传文件的内容类型（MIME类型）
                - transferTo(File dest) -- 将文件下载到指定文件路径
            - 注意事项
                - springboot默认单个文件上传（1MB），多文件上传（10MB），需要更改默认配置，不然会报错
                - 相关配置
                    ```
                        spring:
                            mvc:
                                servlet:
                                    multipart:
                                        max-file-size: 10MB  		# 设置单个文件最大大小为10MB
                                        max-request-size: 100MB  	# 设置多个文件大小为100MB
    
                    ```
            - System类
                - 相关API(这个十分关键)
                    - System.getProperty("user.dir") -- **获取当前你写代码的那个文件目录**，后面跟你想要的文件夹（File.separator+"upload"）
            - File类
                - 相关API
                    - File.separator -- 获取操作系统的分割符(Linux \,Unix /,Windows \or/)



## yuuki's update：10-28

### 1.后端模块化——包与类的设计

直接观察文件结构即可，较为清晰明了。

总开关： [AiVisualApplication.java](src\main\java\com\example\AiVisualApplication.java) 

### 2.训练模型功能实现——食用指南

tmp中都是python服务端的文件。

其中，train.py实现**JA**后端与**py-service**端的交互。

我们可以运行指令`uvicorn train:app --host 0.0.0.0 --port 8000`来开启一个训练函数的**REST API**。

这时我们已经可以用api测试工具进行测试了，测试方案：1.apifox测试；2.谷歌浏览器HackBar插件测试

方法：post方法，访问127.0.0.1:8000，传输json{"a":"1"}（数据传输为参数some_fleld，允许随便传一个别的）



JA后端中， [DemoController.java](src\main\java\com\example\controller\DemoController.java) 开启一个与前端的交互，同时调用一个服务callPythonService，

连接py端apiUrl = "http://127.0.0.1:8000/train"; 

由于技术封锁，目前就简易让java用get方法去请求py端，因为我的数字手写算法训练模型是调用库里的数据集的。

然后运行后端，get访问127.0.0.1:8080/train，即可训练并生成模型在py端服务器。

（这样后端和py端分离在两台服务器上的微服务架构，也可以实现）

整个过程勉强可以实现反馈回显。

over，晚上有点事，润了。