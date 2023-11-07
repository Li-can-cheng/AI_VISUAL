{   
    "task":['Classification'/"Clustering"/"ImageClassification"/"Regression"],---就一个

    "import_data":[
        "import_excel_data"/"import_csv_data"/"import_zip_data"  ---就一个
    ],
    
    "data_preprocessing":[
        {
        "name":"function_name",
        "arguments":{
            "arg1":1,
            "arg2":2
        }
        },
        {
            "name":"function_name",
            "arguments":{
                "arg1":1,
                "arg2":2
            }    
        }

    ],
    
    "model_selection":[
        "MLP":{'linear1':1, "sigmoid":'', 'ReLU':''}
    ],

    "model_evaluation":[
    "F1", "AUC"   ---没有参数
    ]

}