package com.example.yuuki.controller;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class ModelPredictionController {

    @PostMapping("/predict-model")
    public ResponseEntity<String> predictModel(@RequestBody ModelRequest modelRequest) {
        try {
            // 使线程休眠3秒钟（3000毫秒）
            Thread.sleep(3000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt(); // restore interrupted status
            return new ResponseEntity<>("发生错误：线程中断", HttpStatus.INTERNAL_SERVER_ERROR);
        }
        // 这里可以链接数据库并实现具体的预测逻辑
        // 现在只是返回一个固定值 "4"
        return new ResponseEntity<>("结果：4", HttpStatus.OK);
    }

}

class ModelRequest {
    private Long modelId;

    // Getter and setter
    public Long getModelId() {
        return modelId;
    }

    public void setModelId(Long modelId) {
        this.modelId = modelId;
    }
}
