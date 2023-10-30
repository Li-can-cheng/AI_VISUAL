package com.example.ai_visual;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RestController;

import java.io.File;

@RestController
public class HelloController {
    @GetMapping("/hello")
    public String hello() {
        return "Hello, Spring!";
    }

    @Autowired
    private FileUpload file;
//    public static String UPLOAD_DIRECTORY = System.getProperty("user.dir")+ File.separator+"uploads";
    public static String UPLOAD_DIRECTORY = System.getProperty("user.dir")+File.separator+"upload";
//    @PostMapping("/upload")
//    public String getFile(@ModelAttribute MyFile myFile){
//        System.out.println(myFile.getFilename());
//        File dir = new File(UPLOAD_DIRECTORY);
//        if(!dir.exists()){
//            dir.mkdirs();
//        }
//        StringBuffer sb = new StringBuffer(UPLOAD_DIRECTORY + File.separator);
//        sb.append(myFile.getFilename());
//        return myFile.downloadFile(sb.toString());
//        /*String filepath = "C:\\Users\\Lenovo\\Desktop\\FrontTest"+ File.separator+file.getOriginalFilename();
//        File dest = new File(filepath);
//        try {
//            file.transferTo(dest);
//        } catch (IOException e) {
//            throw new RuntimeException(e);
//        }*/
////        Pattern compile = Pattern.compile("\\.[\\w^\\d]+");
////        Matcher matcher = compile.matcher(file.getOriginalFilename());
////        matcher.find();
////        File dest = new File(UPLOAD_DIRECTORY+filename+matcher.group());
////        try {
////            file.transferTo(dest);
////        } catch (IOException e) {
////            throw new RuntimeException(e);
////        }
////        return "文件上传成功！！";
//    }

    @Autowired
    User user;
    @PostMapping("/register")
    public String register(@ModelAttribute User user){
        return user.toString();
    }
}
