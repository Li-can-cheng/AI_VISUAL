package com.example.balabalamodel3;


import lombok.*;
import org.springframework.stereotype.Component;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@ToString
@Component
public class Layer {
    private int linear = -1;
    private String activate_function;
}
