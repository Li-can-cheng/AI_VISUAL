package com.example.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class DemoServiceImpl implements DemoService {
    @Autowired
    DataProcessingService dataProcessingService;
    @Autowired
    ModelTrainingService modelTrainingService;
    @Autowired
    ModelEvaluationService modelEvaluationService;
    @Autowired
    ModelPredictionService modelPredictionService;

    public void preprocessData() {
        dataProcessingService.normalizeImages();
    }

    public void trainModel() {
        modelTrainingService.train();
    }

    public double evaluateModel() {
        return modelEvaluationService.calculateAccuracy();
    }

    public int predict(int imageId) {
        return modelPredictionService.predict(imageId);
    }
}
