package com.example.balabala_test;

import com.google.gson.Gson;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class InfoPostController {
    @Autowired
    private Person person;

    @PostMapping("/sendPerson")
    public String getPersonInfo(@RequestBody Person person){
        Gson gson = new Gson();
        System.out.println(gson.toJson(person));
        return gson.toJson(person);
    }

}
