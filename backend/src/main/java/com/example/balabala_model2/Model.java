package com.example.balabala_model2;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

@Component
public class Model {
    @Autowired
    private List<Command> commands;

    public Model() {
        commands = new ArrayList<>();
    }


    public List<Command> getCommands() {
        return commands;
    }

}
