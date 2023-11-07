package com.example.balabalamodel3;


import lombok.*;
import org.springframework.stereotype.Component;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@ToString
@Component
public class Model {
    private String task;
    private ImportData import_data;
    private DataProcessing[] data_processing;
    private ModelSelectionInterface model_selection;

}
