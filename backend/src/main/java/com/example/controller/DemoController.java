package com.example.controller;

import com.example.service.DemoServiceImpl;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

@RestController
public class DemoController {

    public static String UPLOAD_DIRECTORY = System.getProperty("user.dir") + File.separator + "upload";
    @Autowired
    private DemoServiceImpl demoService;


    //上传文件
    @PostMapping("/upload")
    public ResponseEntity<Map<String, Object>> uploadFile(@RequestParam("file") MultipartFile file) {
        Map<String, Object> response = new HashMap<>();

        File dir = new File(UPLOAD_DIRECTORY);
        if (!dir.exists()) {
            dir.mkdirs();
        }
        try {
            String fileName = file.getOriginalFilename();
            File targetFile = new File(UPLOAD_DIRECTORY + File.separator + fileName);
            file.transferTo(targetFile);

            response.put("message", "文件上传成功！");
            response.put("fileName", fileName);

            return new ResponseEntity<>(response, HttpStatus.OK);
        } catch (Exception e) {
            response.put("message", "文件上传失败！");
            response.put("error", e.getMessage());

            return new ResponseEntity<>(response, HttpStatus.BAD_REQUEST);
        }
    }

    //训练模型
    @PostMapping("/train")
    public String trainModel() {
        demoService.trainModel();
        return "Training service called!";
    }

    //预测模型
    @PostMapping("/predict")
    public String predict(@RequestBody Map<String, Object> request) {
        int imgid = (Integer) request.get("imgid");
        String predict_result = demoService.predict(imgid);
        String res = "Predicting service called!" + "<br>" + predict_result;
        return res;
    }

}
