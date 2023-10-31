package com.example.balabala_model2;

import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PostMapping;

public class Controller {
    @PostMapping("/file")
    public String file(@ModelAttribute){
        return "文件上传成功！！";
    }
}
