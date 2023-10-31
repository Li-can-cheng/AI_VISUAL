package com.example.balabala_model2;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;

public class FileUpload {
    private MultipartFile file;
    final private String base_directory = "Upload";
    private String username;
    private String file_path;
    private String sheet_name;
    private String file_type;
    public FileUpload() {
    }

    public FileUpload(MultipartFile file, String username, String[] file_description) {
        this.file = file;
        this.username = username;
//        for (String item:file_description) {
//            String[] split = item.split("%%%%");
//            this.file_description.put(split[0],split[1]);
//        }
    }

    public void downloadFile(){
        //如果文件路径还没有，那么就自己创建一个文件夹
        this.file_path = System.getProperty("user.dir")+File.separator+base_directory+ File.separator+username;
        File dir = new File(this.file_path);
        if(!dir.exists()){
            dir.mkdirs();
        }
        //下载文件
        this.file_path=this.file_path+File.separator+this.file.getOriginalFilename();
        try {
            this.file.transferTo(new File(this.file_path));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        //判断文件类型
        String[] split = file_path.split("\\.");
        this.file_type = split[split.length-1];
    }

    public MultipartFile getFile() {
        return file;
    }

    public void setFile(MultipartFile file) {
        this.file = file;
    }

    public String getBase_directory() {
        return base_directory;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public String getFile_path() {
        return file_path;
    }

    public void setFile_path(String file_path) {
        this.file_path = file_path;
    }

    public String getSheet_name() {
        return sheet_name;
    }

    public void setSheet_name(String sheet_name) {
        this.sheet_name = sheet_name;
    }

    public String getFile_type() {
        return file_type;
    }

    public void setFile_type(String file_type) {
        this.file_type = file_type;
    }
}
