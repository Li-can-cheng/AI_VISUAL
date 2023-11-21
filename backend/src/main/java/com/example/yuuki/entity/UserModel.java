package com.example.yuuki.entity;

import lombok.Getter;

import javax.persistence.*;
import java.util.Date;

@Getter
@Entity
@Table(name = "user_models")
public class UserModel {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long modelId;

    @Column(nullable = false)
    private Long userId;

    @Column(nullable = false)
    private String modelPath;

    @Column
    private String modelMetadata;

    @Temporal(TemporalType.TIMESTAMP)
    private Date createdAt = new Date();

    // 构造方法、getter和setter省略

    public void setModelId(Long modelId) {
        this.modelId = modelId;
    }

    public void setUserId(Long userId) {
        this.userId = userId;
    }

    public void setModelPath(String modelPath) {
        this.modelPath = modelPath;
    }

    public void setModelMetadata(String modelMetadata) {
        this.modelMetadata = modelMetadata;
    }

    public void setCreatedAt(Date createdAt) {
        this.createdAt = createdAt;
    }
}
