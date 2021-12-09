# Genomic Imputation Using Deep Learning

This project was completed as part of M226 course at UCLA under Professor Sriram Sankararaman.  
Team details:
1) Parth Shettiwar
2) Satvik Mashkaria
3) Parth Patwa  

# Abstract
Genome imputation refers to the statistical inference of unobserved genotypes. It is an
import and and challenging problem in bioinformatics. Most of the previous solutions use statistical
methods like SVD to tackle this problem. In this work we frame the genome imputation problem as
a language generation task and leverage and train the existing deep learning approaches in an autoregressive fashion, to achieve superior results compared to previous approaches. We evaluate our models performance on R2 and accuracy metrics on 1000 Genomes dataset, followed by ablation studies.

Following tables shows illustrates the superiority of our models over existing traditional machine learning algoirthms:

![Accuracy](https://github.com/parth-shettiwar/Genomic_Imputation_using_Deep_Learning/blob/main/Results/Accuracy.png)
![R2](https://github.com/parth-shettiwar/Genomic_Imputation_using_Deep_Learning/blob/main/Results/R2.png)

## Future Work
We plan to modify the existing loss function to traing the model in a better fashion, by putting weight of Mean Allele frequency while computing the cross entropy loss. We also plan to train our models on larger dataset by considering other chromosomes too. Finally, a bidirectional model like BERT would better account for learning the underlying distribution as opposed to uni-directional model
