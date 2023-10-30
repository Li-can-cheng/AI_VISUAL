package com.example.balabala_test;

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
        return person.toString();
    }

}
