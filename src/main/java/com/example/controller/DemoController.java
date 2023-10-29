package com.example.controller;

import com.example.service.DemoServiceImpl;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class DemoController {

    @Autowired
    private DemoServiceImpl demoService;
    @GetMapping("/train")
    public String callPython() {
        demoService.trainModel();
        return "Python service called!";
    }

}
