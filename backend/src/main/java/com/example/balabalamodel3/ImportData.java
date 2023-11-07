package com.example.balabalamodel3;


import lombok.*;
import org.springframework.stereotype.Component;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@ToString
@Component
public class ImportData {
    private String file_path;
    private String method;
}
