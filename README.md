# cs224u_XMTC
Extreme Multi-label Text Classification

## Overview
### Embedding-based method
Extreme multi-label text classification refers to the problem of tagging the most relevant sets of labels to text documents. In XMTC, the number of labels could be extremely large, i.e., thousands or millions. This large number introduces difficulties when solving this problem - data sparsity and computational cost. Many approaches have been introduced to solve these problems in XMTC, which fall into three categories: embedding-based, tree-based, and deep learning method.

### Embedding based method
The main idea of this type of approach is to project labels to a lower-dimensional space. Research focus on different method of compression process (mapping original labels to a lower-dimensional space) and decompression process (mapping back to the original high-dimensional space). Approches include compressed sensoring, bloom filtering, SVD (singular value decomposition), SLEEC (Sparse Local Embeddings for Extreme Multi-label Classification), etc.

### Tree-based method
Similar to decision tree, it partitions the instance space recursively at each non-lead node. However, instead of selecting one feature, it partitions the instance space by hyperplane. FastXML is a representative method in this category.

### Deep learning method
This type of method takes the advantages of taking context information into consideration (while others use bag-of-word). CNN-Kim attempts to apply CNN by concatenate word embeddings of a document, which is analog to an image. XML-CNN adds dynamic max pooling change the objective function s.t. the model is more compatible with XMTC problem.

## Implementation
This project is a python implementation of SLEEC and XML-CNN.

### SLEEC
- Compression process
    - Key idea: SLEEC only preserves pariwise distances between only neareset neighbors of label vectors.
    - Label embedding learning: SVP (Singular Value Projection), non-convex optimization
- Regression
    - Regressor learning: ADMM (Alternating Direction Method of Multipliers)
- Decompression process
    - kNN search of the nearest neighbors of the predicted labels and adds the label vectors of the knn original label vectors
- This [example notebook](presentation/SLEEC_example.ipynb) illustrates an entire workflow of SLEEC with explainations or see [SLEEC code](code/sleec.py).

### XML-CNN
- Network architecture
    - Word embedding inputs
    - Conv layers with different kernel-size
        - Intuiion: `kernel_size=2` looks 2 consecutive words
        - Different kernel-size generates multiple feature maps (analog to image color blobs, edges, etc.)
    - Dynamic max pooling
        - Key idea: since the length of documents are variant, the output volume of the previous conv layer cannot be connected to a dense layer. Dynamic pooling makes sure the consistent output shape and select the most important information.
        - k max pooling & k chunk max pooling
    - Bottleneck FC layer
        - Size reduction and dealing with overfitting
    - FC layer
        - Multi-class: `softmax`, `categorical cross entropy`
        - Multi-label: `sigmoid`, `binary entropy`
- See [XML-CNN code](code/xml_cnn.py).