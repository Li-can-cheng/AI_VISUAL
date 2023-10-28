package com.example.service;

import org.springframework.stereotype.Service;

@Service
public class ModelEvaluationServiceImpl implements ModelEvaluationService {
    public double calculateAccuracy() {
        // 这里返回模型评估结果
        return 0.98;  // 假设准确度为98%
    }
}

