CREATE TABLE user_models (
                             model_id BIGINT AUTO_INCREMENT PRIMARY KEY,
                             user_id BIGINT NOT NULL,
                             model_path VARCHAR(255) NOT NULL,
                             model_metadata TEXT,  -- 存储额外的模型元数据，如JSON格式
                             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                             FOREIGN KEY (user_id) REFERENCES users(id)  -- 外键关联到users表
);
