json_string ='''
{
    "task":["ImageClassification"],    
    "import_data":["import_zip_data",
    "handwriting_digits.zip"],
    "data_preprocessing":[
        {
            "name":"normalize_images",
            "arguments":{
                "mean":""
            }
        },
        {
            "name":"standardize_images",
            "arguments":{
                "mean":""
            }    
        }
    ],
    "model_selection":{
        "name":"MLP",
        "arguments":{
            "epoch":"",
            "layer":{"linear1":256, "sigmoid1":"","linear2":128, "ReLU1":"", "linear3":10, "ReLU2":""}
        }
    }
}'''