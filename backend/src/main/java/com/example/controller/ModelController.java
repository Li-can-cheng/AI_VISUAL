package com.example.controller;

import com.example.balabalamodel3.*;
import com.example.yuuki.UserService;
import com.google.gson.Gson;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.Map;


@RestController
public class ModelController {
    @Autowired
    private Model model;
    @Autowired
    private ImportData import_data;
    @Autowired
    private DataProcessing[] data_processing;
//    @Autowired
//    private ModelSelectionInterface model_selection;
    @Autowired
    private ModelSelectionInterface model_selection;
    @Autowired
    private AcceptData acceptData;
    @Autowired
    Gson gson;

    @Autowired
    private UserService userService;




    @PostMapping("/model/sendFile")
    public String acceptData(@RequestParam String task, @ModelAttribute AcceptData acceptData){
        acceptData.downloadFile();
        import_data.setFile_path(acceptData.getFile_path());
        import_data.setMethod("import_"+acceptData.getFile_type()+"_data");
        model.setTask(task);
        model.setImport_data(import_data);
        System.out.println(gson.toJson(import_data));
        return gson.toJson(import_data);
    }

    @PostMapping("/model/send_data_processing")
    public String acceptDataProcessing(@RequestBody DataProcessing[] data_preprocessing){
        model.setData_preprocessing(data_preprocessing);
        return gson.toJson(data_preprocessing);
    }
    @PostMapping("/model/MLP")
    public ResponseEntity<Map<String, Object>> acceptDataProcessing(@RequestBody MLP model_selection){
//        if("MLP".equals(model_selection.getName())) {
//            mlp.setArguments();
//        }
        model.setModel_selection(model_selection);
        System.out.println(gson.toJson(model));

        // 创建RestTemplate实例
        RestTemplate restTemplate = new RestTemplate();
        // 设置请求头，指定JSON格式(MediaType.APPLICATION_JSON)
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        // 构建请求体对象（这个泛型是自己写的实体类）
        HttpEntity<Model> request = new HttpEntity<>(model, headers);
        // 发送POST请求
        String url = "http://localhost:8000/trainModel";
        ResponseEntity<String> response = restTemplate.postForEntity(url, request, String.class);
        // 处理响应（返回的数据）
        System.out.println(response.getBody());

        // 假设这里得到模型文件路径和元数据
        Long userId = 3L /* 获取用户的ID */;
        String modelPath ="S:\\myJAVA\\Visual-AI-Model-Development-Platform\\python_total_re\\ImageClassification\\model.pth" /* 模型文件的路径，可能从response中解析得到 */;
        String modelMetadata = gson.toJson(model)/* 模型的元数据，可能从model_selection中获取或者其他途径 */;

        // 调用userService的saveModel方法来保存模型信息
        userService.saveModel(userId, modelPath, modelMetadata);

        Map<String, Object> responseData = new HashMap<>();
        responseData.put("result", response.getBody());

        return ResponseEntity.ok(responseData);
    }

    @PostMapping("/model/send_model_evaluation")
    public String acceptModelEvaluation(@RequestBody String[] model_evaluation){
        return null;
    }

    @PostMapping("/test")
    public String test(@RequestBody String json){
        return json;
    }
}
