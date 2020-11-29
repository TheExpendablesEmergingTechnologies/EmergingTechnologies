# Improving Sentiment Analysis with Mutli-Task Learning of Negation

Sentiment Analysis is a process where one can mine people’s opinions from a piece of text. At first glance, this task may look like a simple text classification problem, but once deep dived, one can find out challenges which can affect the sentiment analysis accuracy. For instance, predicting sentiment just using the words in a sentence can lead to major pitfalls such as detecting sarcasm, irony, word ambiguity, use of negating words, also multipolarity. Out of all these phenomena, negation is the most prevalent. Any sentiment analysis model must be able to identify negation and try to remove the effect that its scope has on the final sentiment of a text. In this project, we propose to use Multi-task learning approach, using a cascading and hierarchical neural architecture of LSTM layers. This architecture will explicitly train the model with negation as an auxiliary task and eventually help in improving the main task of sentiment analysis. 

## Model Architecture

![alt text](https://github.com/TheExpendablesEmergingTechnologies/EmergingTechnologies/blob/main/architecture.png)

## Models

- Single-task model
- Multi-task SFU
- Multi-task CD

## Datasets

- [a link](https://nlp.stanford.edu/sentiment/treebank.html)SST
- SemEval 2013 SA task
- SFU Review Corpus
- Conan Doyle Neg (*Sem 2012)
- Streusle Dataset

## Reference: 

[1] Bingel, J. and Søgaard, A. (2017). Identifying beneficial task relations for multi-task learning in deep neural networks. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics, pages 164–169, Valencia, Spain. 

[2] Improving Sentiment Analysis with Multi-task Learning of Negation J E R EM Y B A R N E S , E R I K V E L L D A L , and L I L J A Ø V R E L I D Language Technology Group, University of Oslo email: {jeremycb,erikve,liljao}@ifi.uio.no 
