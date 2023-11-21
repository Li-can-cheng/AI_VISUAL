package com.example.yuuki;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

@Service
public class UserService {

    @Autowired
    private UserModelRepository userModelRepository;

    public void saveModel(Long userId, String modelPath, String modelMetadata) {
        UserModel userModel = new UserModel();
        userModel.setUserId(userId);
        userModel.setModelPath(modelPath);
        userModel.setModelMetadata(modelMetadata);
        userModelRepository.save(userModel);
    }

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private PasswordEncoder passwordEncoder;

    public LoginStatus validateUser(String username, String password) {
        User user = userRepository.findByUsername(username);

        if (user == null) {
            return LoginStatus.USER_NOT_FOUND;
        }

        if (!passwordEncoder.matches(password, user.getPassword())) {
            return LoginStatus.INVALID_PASSWORD;
        }

        return LoginStatus.SUCCESS;
    }


    public boolean registerUser(String username, String password) {
        if (userRepository.findByUsername(username) != null) {
            return false; // 用户名已存在
        }

        User newUser = new User();
        newUser.setUsername(username);
        newUser.setPassword(passwordEncoder.encode(password));
        userRepository.save(newUser);
        return true;
    }
}
