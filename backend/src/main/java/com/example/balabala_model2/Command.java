package com.example.balabala_model2;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

@Component
public class Command {
    private String module;
    @Autowired
    private List<Functions> functions;

    public Command() {
    }

    public Command(String module) {
        this.module = module;
        this.functions = new ArrayList<>();
    }

    public String getModule() {
        return module;
    }

    public void setModule(String module) {
        this.module = module;
    }

    public List<Functions> getFunctions() {
        return functions;
    }

}
