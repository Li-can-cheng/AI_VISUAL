package com.example.balabala_model2;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.HashMap;

@Component
public class Functions {
    private String name;
    private HashMap<String,String> arguments;

    public Functions() {
        this.arguments = new HashMap<>();
    }

    public Functions(String name, HashMap<String,String> arguments) {
        this.name = name;
        this.arguments = arguments;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public HashMap<String,String> getArguments() {
        return arguments;
    }

    public void setArguments(HashMap<String,String> arguments) {
        this.arguments = arguments;
    }
}
