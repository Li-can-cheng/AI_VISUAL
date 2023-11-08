package com.example.balabalamodel3;

import lombok.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@ToString
@Component
public class AcceptData {
    private MultipartFile file;
    private String file_path;
    private String file_type;
    @Value("${FileInput.BaseDirectory}")
    private String base_dir;
    private String username;
    public void downloadFile(){
        //读取properties文件的数据
//        Properties prop = new Properties();
//        InputStream is = AcceptData.class.getResourceAsStream("application.properties");
//        try {
//            prop.load(is);
//        } catch (IOException e) {
//            throw new RuntimeException(e);
//        }
//        System.out.println(base_dir);
        file_path = System.getProperty("user.dir")+ File.separator+"upload"+ File.separator+username;
        //创建文件夹
        File dir = new File(file_path);
        if(!dir.exists()){
            dir.mkdirs();
        }
        //下载文件
        file_path = file_path+File.separator+file.getOriginalFilename();
        try {
            file.transferTo(new File(file_path));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        //获取文件的类型
        analyseFileType();
    }
    private void analyseFileType(){
        Pattern compile = Pattern.compile("\\.[\\w^\\d]+$");
        Matcher matcher = compile.matcher(file.getOriginalFilename());
        matcher.find();
        file_type = matcher.group().split("\\.")[1];

    }
}
