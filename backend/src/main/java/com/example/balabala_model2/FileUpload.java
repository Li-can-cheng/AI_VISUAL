package com.example.balabala_model2;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;

public class FileUpload {
    private MultipartFile file;
    private String filepath;
    @Value("${FileInput.BaseDirectory}")
    private String base_directory;
    private String username;
    private String file_path;
    private String[] file_description;

    public FileUpload() {
    }

    public FileUpload(MultipartFile file, String username, String[] file_description) {
        this.file = file;
        this.username = username;
        this.file_description = file_description;
        //如果文件路径还没有，那么就自己创建一个文件夹
        this.file_path = base_directory+ File.separator+username;
        File dir = new File(this.file_path);
        if(!dir.exists()){
            dir.mkdirs();
        }
        //下载文件

    }


    public MultipartFile getFile() {
        return file;
    }

    public void setFile(MultipartFile file) {
        this.file = file;
    }

    public String getFilepath() {
        return filepath;
    }

    public void setFilepath(String filepath) {
        this.filepath = filepath;
    }

    public String getBase_directory() {
        return base_directory;
    }

    public void setBase_directory(String base_directory) {
        this.base_directory = base_directory;
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

    public String[] getFile_description() {
        return file_description;
    }

    public void setFile_description(String[] file_description) {
        this.file_description = file_description;
    }
}
