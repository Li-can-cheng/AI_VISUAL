数据预处理
{
    "name":"Missing",
    "arguments":{
        "method": --可选--- "mean"/"median"/"interpolate"/"knn"
    }
    }

{
    "name":"Outlines",
    "arguments":{
        "replace_method": --可选-- "extremes"/"mean"/"median"
    }
}

{
    "name":"Filter",
    "arguments":{
        "condition": --必选-- ""
    }
}

{
    "name":"Standardize",
    "arguments":{
        "method": --可选-- "z_score"/"mean_normalization"/"scale_to_unit_length"
    }
}

{
    "name":"Normalize",
    "arguments":{
        "method": --可选-- "min_max"/"max_abs"/"robust"
    }
}

{
    "name":"Similarity",
    "arguments": --无参数-- ""
}
