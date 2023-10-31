package com.example.balabala_test;

import org.springframework.stereotype.Component;

@Component
public class Dog {
    private String name;
    private String allName;
    private String age;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
        this.allName = name+"sh";
    }

    public void setAge(String age) {
        this.age = age;
    }

    public String getAllName(){
        return allName;
    }


    public Dog() {
    }

    public Dog(String name) {
        this.name = name;
    }

    @Override
    public String toString() {
        return "{" +
                "'name'='" + name + '\'' +
                '}';
    }
}
