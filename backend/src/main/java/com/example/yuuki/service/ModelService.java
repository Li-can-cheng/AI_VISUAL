package com.example.yuuki.service;

import com.example.yuuki.entity.UserModel;
import com.example.yuuki.repository.ModelRepository;
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
