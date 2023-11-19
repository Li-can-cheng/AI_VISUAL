package com.example.balabalamodel3;

import lombok.*;
import org.springframework.stereotype.Component;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@ToString
@Component
public class MLPArguments implements ArgumentsInterface{
    private int epochs = -1;
    private Layer[] layers;
}
