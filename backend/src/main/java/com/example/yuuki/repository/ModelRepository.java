package com.example.yuuki.repository;

import com.example.yuuki.entity.UserModel;
import org.springframework.data.jpa.repository.JpaRepository;

public interface ModelRepository extends JpaRepository<UserModel, Long> {
}
