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
    private int linear1 = -1;
    private int sigmoid = -1;
    private int ReLu1 = -1;
    private int linear2 = -1;
    private int ReLu2 = -1;
    private int linear3 = -1;
}
