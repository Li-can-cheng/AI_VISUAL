package com.example.yuuki.controller;

import com.example.yuuki.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
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

    @GetMapping("/register")
    public String showRegisterForm() {
        return "register"; // 返回登录页面的视图名
    }


    @Autowired
    private UserService userService; // 注入UserService

    @PostMapping("/login")
    public String handleLogin(@RequestParam String username, @RequestParam String password, Model model) {
        LoginStatus status = userService.validateUser(username, password);

        switch (status) {
            case USER_NOT_FOUND:
                model.addAttribute("error", "用户名不存在");
                return "login";
            case INVALID_PASSWORD:
                model.addAttribute("error", "密码错误");
                return "login";
            case SUCCESS:
                return "redirect:http://localhost:5173/";
            default:
                model.addAttribute("error", "登录错误，请稍后重试");
                return "login";
        }
    }



    @PostMapping("/register")
    public String handleRegistration(@RequestParam String username, @RequestParam String password, Model model) {
        boolean registrationSuccess = userService.registerUser(username, password);

        if (!registrationSuccess) {
            model.addAttribute("error", "注册失败，请尝试其他用户名");
            return "register";
        }

        // 注册成功，可以重定向到登录页面
        return "redirect:/login";
    }




}
