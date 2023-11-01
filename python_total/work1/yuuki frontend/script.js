
document.addEventListener('DOMContentLoaded', function() {
    // 监听DOMContentLoaded事件，当html文档加载完毕时执行代码
    // 获取UI元素
    const fileInput = document.getElementById('fileInput');
    // const sheetNameInput = document.getElementById('sheetNameInput');
    const epochInput = document.getElementById('epochInput');
    const importButton = document.getElementById('importButton');
    const trainButton = document.getElementById('trainButton');
    const predictButton = document.getElementById('predictButton');
    const dataPreprocessButton = document.getElementById('dataPreprocessButton');
    const readDataButton = document.getElementById('readDataButton');
    const logArea = document.getElementById('log');
    const process_button =document.getElementById('process_button');
    const multiply_factor_input = document.getElementById('multiply_factor_input');
    let tmpfile=null;
    let total_data=null;

    // 更新日志的辅助函数
    function updateLog(message) {
        logArea.value += message + '\n';
    }

    // 导入数据的事件处理器
    importButton.addEventListener('click', function() {
        const file = fileInput.files[0];
        tmpfile = file.name;
        // const sheetName = sheetNameInput.value;
        // 错误处理
        if (!file) {
            updateLog('请先选择一个文件。');
            return;
        }

        // if (!sheetName) {
        //     updateLog('请输入Sheet名。');
        //     return;
        // }

        updateLog('正在发送导入数据请求...');

        // 一些逻辑
        const formData = new FormData();
        formData.append('file', file);

        fetch('http://127.0.0.1:8000/upload', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                console.log('文件上传成功，服务器响应：', data);
                updateLog(JSON.stringify(data));

            })
            .catch(error => console.error('上传文件过程中出问题了：',error));
    });

    // 数据读取
    readDataButton.addEventListener('click', function() {
        updateLog('正在读取数据...');
        // 发送预测请求
        sendCommand("import_data", [{
            name: "import_excel_data",
            arguments: {
                file_path: tmpfile,
                sheet_name: 'Sheet1'
            }
        }]);
    });

    // 数据预处理
    process_button.addEventListener('click', function() {
        const form = document.getElementById('preprocess_form');
        const multiply_factor = parseInt(multiply_factor_input.value, 10) || 1;
        updateLog('正在处理数据...');

        sendCommand("data_preprocessing", [{
            name: "multiply",
            arguments: {
                // data:total_data,
                multiply_factor: multiply_factor
            }
        }]);
    });



    // 开始训练的事件处理器
    trainButton.addEventListener('click', function() {
        const epochs = parseInt(epochInput.value, 10) || 1; // 如果用户没有输入，默认为1

        updateLog('正在发送训练请求...');
        // 发送训练请求
        sendCommand("train", [{
            name: "handwriting_train",
            arguments: {

                input_epochs: epochs
            }
        }]);
    });

    // 开始预测的事件处理器
    predictButton.addEventListener('click', function() {
        updateLog('正在发送预测请求...');
        // 发送预测请求
        sendCommand("predict", [{
            name: "handwriting_predict",
            arguments: {}
        }]);
    });

    // 发送指令到后端的函数
    function sendCommand(module, functions) {
        const url = 'http://127.0.0.1:8000/execute';
        const data = {
            commands: [{
                module: module,
                functions: functions
            }]
        };

        fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(response => {
                updateLog('服务器响应: ' + JSON.stringify(response));
                // total_data = response.data;

            })
            .catch(error => {
                updateLog('请求失败: ' + error.message);
            });
    }
});
