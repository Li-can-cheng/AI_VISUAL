package com.example.controller;

import com.example.service.PythonCallerService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class DemoController {

    @Autowired
    private PythonCallerService pythonCallerService;

    @GetMapping("/train")
    public String callPython() {
        pythonCallerService.callPythonService();
        return "Python service called!";
    }
}
