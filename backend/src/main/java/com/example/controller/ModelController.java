package com.example.controller;


import com.example.balabalamodel3.*;
import com.google.gson.Gson;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

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
    private MLP model_selection;
    @Autowired
    private AcceptData acceptData;
    @Autowired
    Gson gson;


    @PostMapping("/model/sendFile")
    public String acceptData(@RequestParam String task, @ModelAttribute AcceptData acceptData){
        acceptData.downloadFile();
        import_data.setFile_path(acceptData.getFile_path());
        import_data.setMethod("import_"+acceptData.getFile_type()+"_data");
        model.setTask(task);
        model.setImport_data(import_data);
        return gson.toJson(import_data);
    }

    @PostMapping("/model/send_data_processing")
    public String acceptDataProcessing(@RequestBody DataProcessing[] data_processing){
        model.setData_processing(data_processing);
        return gson.toJson(data_processing);
    }
    @PostMapping("/model/send_model_selection")
    public String acceptDataProcessing(@RequestBody MLP model_selection){
//        if("MLP".equals(model_selection.getName())) {
//            mlp.setArguments();
//        }
        model.setModel_selection(model_selection);

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
        return response.getBody();
    }
}
