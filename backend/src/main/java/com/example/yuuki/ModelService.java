package com.example.yuuki;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class ModelService {
    @Autowired
    private ModelRepository modelRepository;

    public List<UserModel> getAllModels() {
        return modelRepository.findAll();
    }
}
