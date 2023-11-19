package com.example.yuuki;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

@Controller
public class UserController {

    @GetMapping("/login")
    public String showLoginForm() {
        return "login"; // 返回登录页面的视图名
    }


    @PostMapping("/login")
    public String handleLogin(@RequestParam String username, @RequestParam String password, Model model) {
        // TODO: 实际的登录逻辑
        boolean loginSuccess = false; // 假设登录结果

        if (!loginSuccess) {
            model.addAttribute("error", "无效的用户名或密码");
            return "login";
        }

        // 登录成功的处理，比如重定向到主页
        return "redirect:/home";
    }

    @GetMapping("/register")
    public String showRegistrationForm() {
        return "register";
    }



}
