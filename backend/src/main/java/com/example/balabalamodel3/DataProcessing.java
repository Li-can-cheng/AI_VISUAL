package com.example.balabalamodel3;

import lombok.*;
import org.springframework.stereotype.Component;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@ToString
@Component
public class DataProcessing {
    private String name;
    private String arguments;
}
