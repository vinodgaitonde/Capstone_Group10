# Next Word Prediction Model


Next Word Prediction is a widely used feature in smartphone keyboards, messaging applications, and search engines such as Google. It enhances user experience by predicting and suggesting the next word in a sentence based on the previous input. This abstract presents a deep learning approach to building a Next Word Prediction model using Python, leveraging the TensorFlow and Keras libraries. We train a Recurrent Neural Network (RNN), specifically a Long Short-Term Memory (LSTM) network, which is well-suited for handling sequential data and long-term dependencies, making it ideal for this task.

The model is trained on a text corpus from a Sherlock Holmes novel, which serves as a representative dataset for constructing language models. The key steps in building the model involve collecting a diverse dataset, preprocessing the text data by cleaning and tokenizing it, and preparing input-output pairs where the model predicts the next word based on the sequence of preceding words.

Feature engineering is done through word embeddings, which capture semantic relationships between words, enabling the model to understand the context in which words are used. The LSTM model is trained by optimizing various hyperparameters, including the number of layers, units, batch size, and learning rate, to maximize its predictive accuracy.

Throughout the training process, techniques such as dropout regularization and early stopping are applied to prevent overfitting and ensure the model generalizes well to unseen text. After training, the model is evaluated on its ability to predict the next word in various test sequences, and fine-tuning is performed to further improve accuracy. The LSTM's ability to retain and utilize long-term dependencies in text sequences makes it a powerful tool for next word prediction.

# Next Word Prediction Workflow
1) Data Collection: Text from a Sherlock Holmes novel serves as the dataset.
2) Preprocessing: Text cleaning and tokenization into sequences of words.
3) Word Embedding: Transformation of words into vector representations.
4) LSTM Model: Sequential learning with a Long Short-Term Memory network.
5) Prediction Output: The trained model predicts the next word in a sequence.

![image](https://github.com/user-attachments/assets/0a3a57b3-7c1a-4ec9-a9f5-76cb010f7dec)

# LSTM Architecture for Next Word Prediction
1) Input Layer: A sequence of words is fed into the model.
2) Embedding Layer: Words are converted into dense vectors (embeddings) that represent their semantic meaning.
3) LSTM Cells: Each LSTM cell processes the input, retaining information about previous inputs through internal memory and hidden states.
4) Output Layer: The model predicts the next word in the sequence based on the learned patterns.

In practical applications, this model mimics the behavior of predictive text systems in smartphones and other applications. Google’s search engine, for instance, makes use of a similar approach, drawing on a user’s browsing history and previous search inputs to predict the next word more accurately. The implementation is done in Python, and the results demonstrate the effectiveness of deep learning models, particularly LSTMs, in predicting the next word in a sequence.

--------------------------------------------------------------------------------------------------------------------------------------------------------
# Implementation 

# Data Set Info
Source : https://www.gutenberg.org/
         
         The Adventures of Sherlock Holmes, by Arthur Conan Doyle 

# Data Clean up & Processing
    UTF-8 encoded “.txt” file of the novel is used.
    Tokenize & create unique word dictionary.
    Create n-gram.
    Pad sequences to make them of same length.
    Prepare features(n-gram) against label.
    Convert output array into suitable format for training
    
# ML Methodologies Used
Recurrent Neural Network (RNN)

    RNN model is built using Keras library.
    The Sequential model is created which is a linear stack of layers in Keras.
    Layers in the model
   ![image](https://github.com/user-attachments/assets/e8a48fbf-d132-4aaa-a0de-68fd31b50785)

  ![image](https://github.com/user-attachments/assets/933f3091-b0e3-47a2-9501-2f16542427ac)

# Docker Container
  ![image](https://github.com/user-attachments/assets/9064a972-96fe-4258-b703-e33fc7cc8515)

  ![image](https://github.com/user-attachments/assets/88837fc9-c9c6-4928-8824-b008011ca362)


# Model Tuning

![image](https://github.com/user-attachments/assets/af508aed-6aac-40cf-b1ba-6b37724990f7)

# Predicted Words Sample

![image](https://github.com/user-attachments/assets/e992382d-1f37-441a-bbbf-149bb8ca5cd8)

# Conclusion

![image](https://github.com/user-attachments/assets/cdcb4b76-67ea-4bc4-9ecd-72e2e778550e)

# Link to Source Code & Docker


# Future Enhancements
![image](https://github.com/user-attachments/assets/f4a2a0a0-6e04-4847-9d07-f271012b709a)


  



   

    
    





    





