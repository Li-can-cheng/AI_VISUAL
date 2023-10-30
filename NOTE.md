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
        ```yaml
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

## balabala'update 2023/10/30  实现了  front-->java_server-->python_script 的数据传输
### 先验知识
- post传输的数据格式
  - json格式
    - 前端可能需要ajax实现
    - java端直接用@RequestBody完成对象映射
    - py端用fastapi的Body模块实现数据映射
  - multipart/form-data格式
    - 前端用form表单实现（注意enctpye="multipart/form-data",这种可以实现文件的传输）
    - java端用@ModelAtribute完成数据映射
    - py端用fastapi的Form模块实现数据映射（键名要与发送的**数据的键名一一对应**）
  - 此处应该插入一张图片（但是我太懒了）
### 具体实现方式
  - java端
    - 接收数据
      1. 创建实体类
        ```java
            public class TestPostFastAPI {
                  private String data;
                  //省略构造函数和getter&setter函数
              }
        ```
      2. post请求数据接收（这边用form表单形式接收）
        ```java
            //@PostMapping("/predict")
            public String callPythonToPredict(@ModelAttribute TestPostFastAPI tpfa){
                System.out.println(tpfa.getData());
                return pythonCallerService.pythonPredictService(tpfa);
            }
        ```
      3. 发送post请求（这边用json形式实现）
        ```java
           public String pythonPredictService(TestPostFastAPI tpfa){
                // 创建RestTemplate实例
                RestTemplate restTemplate = new RestTemplate();
                // 设置请求头，指定JSON格式(MediaType.APPLICATION_JSON)
                HttpHeaders headers = new HttpHeaders();
                headers.setContentType(MediaType.APPLICATION_JSON);
                // 构建请求体对象（这个泛型是自己写的实体类）
                HttpEntity<TestPostFastAPI> request = new HttpEntity<>(tpfa, headers);
                // 发送POST请求
                String url = "http://localhost:8000/testJson";
                ResponseEntity<String> response = restTemplate.postForEntity(url, request, String.class);
                // 处理响应（返回的数据）
                return response.getBody();
           }
        ```
  - py端实现
    1. form形式的接收数据
       ```python
           from fastapi import Body
           #这个Body可以解析JSON格式的post请求体数据
           @app.post("/testJson")
           def test(data = Body(None)):
               print(data)
               return {"result":data}
       ```
    2. json形式的接收数据
        ```python
           from fastapi import Form
           #这个Body可以解析Form表单格式的post请求体数据
           @app.post("/testForm")
           def test(data = Form(None)):
               print(type(data))
               return {"result":data}
       ```      