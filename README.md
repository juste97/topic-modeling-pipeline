# Topic Modeling Pipeline

This repository contains a pipeline for topic modeling using BERTopic, a Python library for topic modeling with BERT embeddings. The pipeline includes preprocessing, tokenization, embedding generation, dimensionality reduction, clustering, and visualization.


**Note: This is my first attempt at creating a class-based pipeline. If you have suggestions or best practices to share, please let me know! I'd greatly appreciate any feedback 😊**

## Features

- **Preprocessing**: Clean and sample Twitter data from a dataframe.
- **Tokenization**: Tokenize the documents and build a vocabulary.
- **Embedding Generation**: Encode the documents using SentenceTransformer. You can also load previously generated embeddings to speed up experiments.
- **Dimensionality Reduction**: Reduce the dimensionality of the embeddings using UMAP. Previously reduced embeddings can be loaded for faster processing.
- **Clustering**: Cluster the reduced embeddings using HDBSCAN. You can also load previously generated clusters to speed up experiments.
- **Visualization**: Visualize the clusters and generate word clouds for each topic.



Specify paths for embeddings and reduced embeddings to expedite experimentation. 


The pipeline saves everything (including cluster plot and topics overview) in a unique folder every time an instance is created.



## Classes

1. **Dimensionality**: A dummy class to allow skipping clustering during model fitting.
2. **Plots**: A class to visualize topic modeling related information.
3. **Preprocessor**: A pipeline to clean and sample Twitter data from a dataframe.
4. **TopicModelPipeline**: The main class that integrates all the functionalities and provides an end-to-end pipeline for topic modeling.

## How to Use

1. **Import**:
 ```python
 from src.topic_model import TopicModelPipeline
```

1.5 **Optional: Overwrite class for cleaning Tweets**
```python
from src.preprocessor import Preprocessor

def new_clean_text_columns(self):
    """
    Method to overwrite clean_text_columns method from Preprocessor.
    """
    self.df[self.text_column] = "Example"
    
Preprocessor.clean_text_columns = new_clean_text_columns
```


2. **Initialization**:
```python
pipeline = TopicModelPipeline(
   project_name="Your_Project_Name",
   path="path/to/save/output",
   docs_path="path/to/documents",
   file_type="csv",
   model="all-MiniLM-L6-v2",
   time_column="timestamp_column_name",
   text_column="text_column_name",
   sample=True,
   sample_frequency="day",
   clean_text=True
)
```

3. **Visualize Clusters:**
 ```python
 pipeline.plot_clusters()
```


4. **Visualize Word Clouds:**
 ```python
 pipeline.plot_wordclouds()
```

## GPU acceleration

For GPU acceleration switch out HDBSCAN and UMAP import with
 ```python
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN
```
if RAPIDS cuML installation is possible.

## Dependencies
matplotlib
seaborn
wordcloud
adjustText
pandas
numpy
nltk
sklearn
sentence_transformers
bertopic
torch
umap
hdbscan
transformers
huggingface_hub
joblib
tqdm
swifter
