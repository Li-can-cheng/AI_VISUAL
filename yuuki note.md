## 10-30 yuuki项目更新Note

### 如何运行项目

1. **前端运行**

    port:8081

    ```bash
    cd frontend
    npm run serve
    ```

2. **后端运行**

    port:8080

    运行 [AiVisualApplication.java](..\..\myJAVA\Visual-AI-Model-Development-Platform\backend\src\main\java\com\example\AiVisualApplication.java) 即可

3. **py端运行**

    port:8000

    ```
    cd tmp
    
    uvicorn main:app --host 127.0.0.1 --port 8000
    ```

    

### API 的基本使用

1. **上传文件**

    ```http
    POST /upload
    ```

    | 参数 | 类型 | 描述 |
    | ---- | ---- | ---- |
    |      |      |      |

    **响应**

    ```json
    文件上传成功！
    ```

2. **训练模型**

    ```http
    POST /train
    ```

    | 参数 | 类型 | 描述 |
    | ---- | ---- | ---- |
    |      |      |      |

    **响应**

    ```json
    Training service called!
    ```

3. **训练模型**

    ```http
    POST /predict
    ```

    | 参数  | 类型 | 描述        |
    | ----- | ---- | ----------- |
    | imgid | json | {"imgid":4} |

    **响应**

    ```json
    Predicting service called!
    Response from model: {"result":4}
    ```



## 10-31凌晨

##### Y_fileinfo——model

一个文件对象，包含要传递的参数。

```java
package com.example.model;

public class Y_fileInfo {
    private String path;
    private String name;
    private String type;
    private String id;
    private String user;
    private long size;
    //getters and setters
}
```



###### Y_ctl

get访问/1实现传参，手动填充参数。

```java
package com.example.controller;

import com.example.model.Y_fileInfo;
import com.example.service.yuuki_serv;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController  // 注解是必需的，以便Spring Boot识别它为一个REST控制器
public class Y_ctl {

    @Autowired
    private Y_serv serv;

    @GetMapping("/1")
    public String hello() {
        Y_fileInfo fileInfo = new Y_fileInfo();  
        // 不能在类的属性部分执行逻辑操作
        
        fileInfo.setPath("1");
        fileInfo.setName("your_file_name");
        fileInfo.setType("your_file_type");
        fileInfo.setId("your_file_id");
        fileInfo.setUser("your_user_id");
        fileInfo.setSize(1); 
        // 注意这里应和Y_fileInfo里的setSize接受的参数类型一致

        serv.task1(fileInfo);  // 将fileInfo对象作为参数传递
        return "Successfully!";
    }
}


```



###### Y_serv

传参操作

```java
package com.example.service;
import com.example.model.Y_fileInfo;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

@Service
public class Y_serv {

    public static void task1(Y_fileInfo fileInfo) {
        RestTemplate restTemplate = new RestTemplate();
        String pythonApiUrl = "http://127.0.0.1:8000/hi";
        restTemplate.postForObject(pythonApiUrl, fileInfo, String.class);
    }

}

```

