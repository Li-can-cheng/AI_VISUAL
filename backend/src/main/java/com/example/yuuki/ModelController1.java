package com.example.yuuki;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

import java.util.List;

@Controller
public class ModelController1 {

    private final UserModelRepository userModelRepository;

    @Autowired
    public ModelController1(UserModelRepository userModelRepository) {
        this.userModelRepository = userModelRepository;
    }
    @GetMapping("/")
    public String index() {
        return "index"; // 这里的 "index" 是Thymeleaf模板的名称，确保它与您的html文件名匹配
    }
    @GetMapping("/model-prediction")
    public String showModelPrediction(Model model) {
        List<UserModel> allUserModels = userModelRepository.findAll();
        System.out.println("Retrieved models: " + allUserModels); // 临时日志
        model.addAttribute("models", allUserModels);
        return "predict";  // 返回 Thymeleaf 模板的名称
    }
}
