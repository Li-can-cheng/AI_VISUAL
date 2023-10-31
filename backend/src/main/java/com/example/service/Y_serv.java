package com.example.service;
import com.example.model.Y_fileInfo;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

@Service
public class Y_serv {

    public static void task1(Y_fileInfo fileInfo) {
        RestTemplate restTemplate = new RestTemplate();
        String pythonApiUrl = "http://127.0.0.1:8000/hi";
        restTemplate.postForObject(pythonApiUrl, fileInfo, String.class);
    }

}
