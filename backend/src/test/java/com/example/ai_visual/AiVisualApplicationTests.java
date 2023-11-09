package com.example.ai_visual;

import com.example.balabalamodel3.AcceptData;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.test.context.SpringBootTest;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

@SpringBootTest
class AiVisualApplicationTests {

    @Value("${FileInput.BaseDirectory}")
    private String file;
    @Test
    public void contextLoads() {
        System.out.println(file);
    }


    @Autowired
    private School school;
    @Autowired
    private Student student;
    @Test
    public void testPerson(){
        student.setAge(1);
        school.setPerson(student);
        System.out.println(school);
    }

}
