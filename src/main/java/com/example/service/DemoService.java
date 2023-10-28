package com.example.service;

public interface DemoService {
    void preprocessData();
    void trainModel();
    double evaluateModel();
    int predict(int imageId);
}
