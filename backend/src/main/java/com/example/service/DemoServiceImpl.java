package com.example.service;

import com.example.model.RequestObject;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

@Service
public class DemoServiceImpl implements DemoService {

    @Autowired
    private RestTemplate restTemplate;
    private MultipartFile file;
    private String filename;

    //预处理
    public void preprocessData() {

    }

    //训练模型
    public void trainModel() {
        // Python服务端API的URL
        String apiUrl = "http://127.0.0.1:8000/train";

        //创建请求对象,该对象里一般是存着请求的参数
        RequestObject requestBody = new RequestObject();

        //设置请求参数value值为my_value
        requestBody.setValue("my_value");

        //看看自己发了啥子字段
        System.out.println(requestBody.getValue());

        //万事俱备，post请求发送一个json到训练api（接受转换成String类）
        ResponseEntity<String> response = restTemplate.postForEntity(apiUrl, requestBody, String.class);

        //处理响应
        if (response.getStatusCode().is2xxSuccessful()) {
            System.out.println("Success: " + response.getBody());
        } else {
            System.out.println("Failed: " + response.getStatusCode());
        }
    }

    //模型评估
    public double evaluateModel() {
        return 0;
    }

    //预测图片
    public String predict(int imageId) {
        String url = "http://127.0.0.1:8000/predict/";
        double[][] my2DArray = readAndConvertImage(imageId);  //使用一个函数将图片转换成二维数组

        Map<String, Object> map = new HashMap<>();
        //请求参数
        map.put("data", my2DArray);

        //请求头
        HttpHeaders headers = new HttpHeaders();
        //json类型
        headers.setContentType(MediaType.APPLICATION_JSON);
        //
        HttpEntity<Map<String, Object>> request = new HttpEntity<>(map, headers);
        ResponseEntity<String> response = restTemplate.postForEntity(url, request, String.class);

        return "Response from model: " + response.getBody();
    }


    public double[][] readAndConvertImage(int imageId) {
        // 根据 imageId 从某个路径读取图片，这里假设图片名是 imageId.jpg
        //TODO:绝对路径改成相对路径
        String imagePath = "S:\\myJAVA\\Visual-AI-Model-Development-Platform\\upload\\" + imageId + ".jpg";
        System.out.println(imagePath);

        BufferedImage img;
        try {
            img = ImageIO.read(new File(imagePath));
        } catch (IOException e) {
            e.printStackTrace();
            return null;  // 或者其他错误处理
        }

        int width = img.getWidth();
        int height = img.getHeight();
        double[][] imageArray = new double[height][width];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixel = img.getRGB(x, y);
                int r = (pixel >> 16) & 0xff;
                int g = (pixel >> 8) & 0xff;
                int b = pixel & 0xff;
                double gray = 0.299 * r + 0.587 * g + 0.114 * b;
                imageArray[y][x] = gray;
            }
        }

        return imageArray;
    }


}
