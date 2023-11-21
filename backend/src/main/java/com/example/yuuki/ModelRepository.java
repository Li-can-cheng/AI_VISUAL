package com.example.yuuki;

import org.springframework.data.jpa.repository.JpaRepository;

public interface ModelRepository extends JpaRepository<UserModel, Long> {
}
