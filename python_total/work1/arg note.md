## main.py

restapi的分配枢纽，目前只有一个router分配给了total_excution.py



## total_excution.py

主枢纽

```
@router.post("/execute")
```



##  import_data.py 



##### 

##### import_csv_data

导入csv

```python
def import_csv_data(file_path):
    return data
```

输入类型 str

返回类型 pandas.dataframe



##### 

##### import_excel_data

导入excel

```python
def import_excel_data(file_path, sheet_name=None):
    return data
```

输入类型 str str(optional)

返回类型 pandas.dataframe



##### read_image

读图

```python
def read_image(image_path):
    return image
```

输入类型 str str(optional)

返回类型 numpy.ndarray

##  data_preprocessing.py 



##### multiply

乘法

```python
def multiply(data, multiply_factor: int):
```

输入类型 dataframe,int

返回类型 dataframe





```
def handle_missing_values1(data):
```

```
def handle_missing_values2(data):
```

```
def handle_outliers1(data, method='z-score', threshold=3):
```

```
def handle_outliers2(data, method='mean', threshold=3):
```

```
def filter_data(data, condition):
```

```
def modify_data(data, operation, new_data=None):
```

```
def merge_data(data1, data2, on=None):
```

```
def split_dataset(data, test_size=0.2, random_state=None):
```

```
def tokenize_text(text):
```

```
def segment_image(image_path):
```

```
def convert_to_gray(image_path):
```

## train.py



```
def handwriting_train(input_epochs):
```



## predict.py

```
def handwriting_predict():
```