package com.example.balabala_model2;

import com.google.gson.Gson;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.io.File;

@RestController
public class Controller {

    @Autowired
    private Model model;
    @Autowired
    private Functions[] function;
    @Autowired
    private Command[] command;
    private Gson gson = new Gson();
    @PostMapping("/file")
    public String file(@ModelAttribute FileUpload fileUpload){
        fileUpload.downloadFile();
        if("xlsx".equals(fileUpload.getFile_type())){
            function[0].setName("import_excel_data");
            function[0].getArguments().put("file_path",fileUpload.getFile_path());
            function[0].getArguments().put("sheet_name",fileUpload.getSheet_name());
        }
        command[0].setModule("import_data");
        System.out.println(fileUpload.getFile_type());
        return gson.toJson(model);
    }
    @PostMapping("/data_process")
    public String data_process(@RequestBody InputDataProcess inputDataProcess){
        function[1].setName(inputDataProcess.getName());
        function[1].getArguments().put("multiply_factor",inputDataProcess.getMultiply_factor());

        command[1].setModule("data_processing");
        command[1].getFunctions().add(function[1]);
        model.getCommands().add(command[1]);

        return gson.toJson(model);
    }
}
