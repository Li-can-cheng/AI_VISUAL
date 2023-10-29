package com.example.service;

import com.example.model.RequestObject;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

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
    @Autowired
    private RestTemplate restTemplate;



    public void preprocessData() {
        dataProcessingService.normalizeImages();
    }

    public void trainModel() {
        String apiUrl = "http://127.0.0.1:8000/train"; // 替换为Python服务端API的真实URL
        RequestObject requestBody = new RequestObject();
        requestBody.setSomeField("someValue");

        ResponseEntity<String> response = restTemplate.postForEntity(apiUrl, requestBody, String.class);

        // 以下是处理响应的代码，可以根据你的需要进行更改
        if (response.getStatusCode().is2xxSuccessful()) {
            System.out.println("Success: " + response.getBody());
        } else {
            System.out.println("Failed: " + response.getStatusCode());
        }
    }

    public double evaluateModel() {
        return modelEvaluationService.calculateAccuracy();
    }

    public int predict(int imageId) {
        return modelPredictionService.predict(imageId);
    }
}
