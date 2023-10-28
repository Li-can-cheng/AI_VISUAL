package com.example.controller;

import com.example.service.DemoService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/handwriting")
public class DemoController {
    @Autowired
    DemoService handwritingService;

    @PostMapping("/preprocess")
    public ResponseEntity<String> preprocess() {
        handwritingService.preprocessData();
        return new ResponseEntity<>("Data Preprocessed Successfully", HttpStatus.OK);
    }

    @PostMapping("/train")
    public ResponseEntity<String> train() {
        handwritingService.trainModel();
        return new ResponseEntity<>("Model Trained Successfully", HttpStatus.OK);
    }

    @GetMapping("/evaluate")
    public ResponseEntity<Double> evaluate() {
        double accuracy = handwritingService.evaluateModel();
        return new ResponseEntity<>(accuracy, HttpStatus.OK);
    }

    @GetMapping("/predict/{id}")
    public ResponseEntity<Integer> predict(@PathVariable int id) {
        int result = handwritingService.predict(id);
        return new ResponseEntity<>(result, HttpStatus.OK);
    }
}
