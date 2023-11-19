import { defineStore } from "pinia";

export const useDataStore = defineStore("data", {
    state: () => {
        return {
            data: {
                file_input: {
                    file: null,
                    username: "balabala"
                },
                data_processing: [

                ],
                model_selection: {
                    name: "MLP",
                    model_evaluation: ["Accuracy","F1"],
                    arguments: {
                        epochs: 100,


                        layers: [
                            {
                                linear:128,
                                activate_function:"ReLU"
                            },
                            {
                                linear:128,
                                activate_function:"ReLU"
                            },
                            {
                              linear:10,
                              activate_function: "Softmax"
                            }
                        ]
                    }
                }
            },
            current_data_processing: {
                name: "",
                arguments: {
                    mean: ""
                }
            },
            current_select:""
        }
    },
    getters: {

    },
    actions: {
        set_file_input(file_input) {
            if (file_input.username) this.data.file_input.username = file_input.username;
            if (file_input.file) this.data.file_input.file = file_input.file;
        },
        get_file_input() {
            return this.data.file_input;
        },
        // 操作数据预处理的函数
        add_data_processing() {
            //如果current_data_processing没有被赋值则返回
            if (this.current_data_processing.name === "") return;

            
            this.data.data_processing.push({
                name: this.current_data_processing.name,
                arguments: {
                    mean: this.current_data_processing.arguments.mean
                }
            });
            this.current_data_processing = {
                name: "",
                arguments: {
                    mean: ""
                }
            }
            console.log(this.data.data_processing);
        },
        delete_data_processing(name) {
            this.data.data_processing = this.data.data_processing.filter(item => {
                return name !== item.name;
            })
        },
        get_data_processing() {
            return this.data.data_processing;
        },
        set_current_data_processing(current_data_processing) {
            if (current_data_processing.name) this.current_data_processing.name = current_data_processing.name;
            if (current_data_processing.arguments.mean)
                this.current_data_processing.arguments.mean = current_data_processing.arguments.mean;
        },
        get_current_data_processing() {
            return this.current_data_processing;
        },
        set_current_select(current_select){
            this.current_select = current_select;
        },
        get_current_select(){
            return this.current_select;
        },
        get_model_selection(){
            return this.data.model_selection;
        }
    }
})