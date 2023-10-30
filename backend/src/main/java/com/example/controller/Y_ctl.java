package com.example.controller;

import com.example.model.Y_fileInfo;
import com.example.service.Y_serv;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController  // 注解是必需的，以便Spring Boot识别它为一个REST控制器
public class Y_ctl {

    @Autowired
    private Y_serv serv;

    @GetMapping("/1")
    public String hello() {
        Y_fileInfo fileInfo = new Y_fileInfo();  // 移到方法内部，不能在类的属性部分执行逻辑操作
        fileInfo.setPath("1");
        fileInfo.setName("your_file_name");
        fileInfo.setType("your_file_type");
        fileInfo.setId("your_file_id");
        fileInfo.setUser("your_user_id");
        fileInfo.setSize(1);  // 注意这里应和你的Y_fileInfo里的setSize接受的参数类型一致

        serv.task1(fileInfo);  // 将fileInfo对象作为参数传递
        return "Successfully!";
    }
}

