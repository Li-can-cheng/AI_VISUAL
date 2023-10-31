package com.example.balabala_model2;

public class InputDataProcess {
    private String name;
    private String multiply_factor;

    public InputDataProcess() {
    }

    public InputDataProcess(String name, String multiply_factor) {
        this.name = name;
        this.multiply_factor = multiply_factor;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getMultiply_factor() {
        return multiply_factor;
    }

    public void setMultiply_factor(String multiply_factor) {
        this.multiply_factor = multiply_factor;
    }
}
