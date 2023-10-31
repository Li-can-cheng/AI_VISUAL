package com.example.balabala_test;


import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.HashMap;

@Component
public class Person {
    private String name;
//    @Autowired
//    private Dog dog;
    private String[] scores;

    public String[] getScores() {
        return scores;
    }

    public void setScores(String[] scores) {
        this.scores = scores;
    }

    public Person() {
    }

    public Person(String name, Dog dog) {
        this.name = name;
//        this.dog = dog;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

//    public Dog getDog() {
//        return dog;
//    }
//
//    public void setDog(Dog dog) {
//        this.dog = dog;
//    }

}
