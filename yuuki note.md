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

4. 

