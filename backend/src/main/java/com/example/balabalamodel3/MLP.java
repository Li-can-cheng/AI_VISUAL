package com.example.balabalamodel3;

import lombok.*;
import org.springframework.stereotype.Component;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@ToString
@Component
public class MLP implements ModelSelectionInterface{
    final private String name = "MLP";
    private String[] model_evaluation;
    private MLPArguments arguments;
}
