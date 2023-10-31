package com.example.balabala_model.input_data;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

/**
 * 文件输入的类
 *
 */
@Component
public class InputFileData {
    private MultipartFile file;
    @Value("${FileInput.BaseDirectory}")
    static private String base_directory;
    private String username;
    private String filepath;
    private String filename;

    private String file_type;//判断是压缩包还是单个文件
//    private String step_name;

    public InputFileData() {
    }

    public InputFileData(MultipartFile file, String username, String file_type) {
        this.file = file;
        this.username = username;
        this.file_type = file_type;
    }

    public String DownloadFile(){
        return null;
    }

    public String getFilepath() {
        return filepath;
    }

    public void setFilepath(String filepath) {
        this.filepath = filepath;
    }

    public String getFile_type() {
        return file_type;
    }

    public void setFile_type(String file_type) {
        this.file_type = file_type;
    }

    public String getFilename() {
        return filename;
    }

    public void setFilename(String filename) {
        this.filename = filename;
    }

    @Override
    public String toString() {
        return "InputFileData{" +
                "file=" + file +
                ", username='" + username + '\'' +
                ", filepath='" + filepath + '\'' +
                ", filename='" + filename + '\'' +
                '}';
    }
}
