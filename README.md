# AttentionRank
Repository to develop AttentionRank algorithm as library



Based on the works: https://github.com/hd10-iupui/AttentionRank and https://github.com/oeg-upm/AttentionRankLib

  


## Install

Using Python 3.9


```
pip install -r requirements.txt
```


```
pip install -e .
```

```
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm

```


## Execution

```
python main.py --dataset_name example --model_name_or_path PlanTL-GOB-ES/roberta-base-bne --model_type roberta --lang es --type_execution exec --k_value 15
```
Important: The dataset must be in this folder with any name and all documents must be inside another folder named docsutf8
Results will be provided inside the folder of the dataset in a folder named res+k_value

### Variables
--dataset_name: folder where the dataset is. Must be in the root folder of this project. (to be improved)  
--model_name_or_path PlanTL: model for the system   
--model_type roberta: bert/roberta for the different tokenizers  
--lang: language en/es  
--type_execution: exec/eval to perform or not evaluation at the end  
--k_value: nº of top keyphrases  

## Docker run 
For a fast run use the dockerfile and this two commands. 

```
docker build -t attentionranklib .

``` 

```
docker run --rm -v ./example:/app/example attentionranklib --dataset_name example --model_name_or_path PlanTL-GOB-ES/roberta-base-bne --model_type roberta --lang es --type_execution exec --k_value 15
```



## Acknowledgments 

Este código se ha mejorado y adaptado en el marco del proyecto TeresIA, proyecto de investigación financiado con fondos de la Unión Europea Next GenerationEU / PRTR a través del Ministerio de Asuntos Económicos y Transformación Digital (hoy Ministerio para la Transformación Digital y de la Función Pública). 


