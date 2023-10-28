package com.example.service;

import org.springframework.stereotype.Service;

@Service
public class ModelPredictionServiceImpl implements ModelPredictionService {
    public int predict(int imageId) {
        // 这里实现模型预测逻辑
        return 5;  // 假设识别为数字5
    }
}

