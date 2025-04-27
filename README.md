# Google 2B: Search Query Recommendation System

[![Cool Google GIF](http://googleusercontent.com/image_generation_content/0)](http://googleusercontent.com/image_generation_content/0)

**Purpose**: This project was designed to cultivate a deep understanding of the data and technical skills necessary to build a robust search engine, similar to Google’s. Our primary objective was to develop an LLM capable of replicating the predictive autocomplete functionality found in modern search engines, built entirely from the ground up.

Explore our presentation slides [here](https://docs.google.com/presentation/d/1q_QX90_682fVRCP2913vI9J4t7gihrB5tEJYidMY4js/edit?usp=sharing)!

## Meet the Team

**Team Members**: Viswaretas Kotra

**Challenge Advisors & TAs**: Josh McAdams, Esther Lou

## Tech Stack

* Hugging Face
* Google Colab
* Pandas
* PyTorch
* TikToken
* Visual Studio Code

## Methodology & Implementation

* **Dataset**: Sourced from Hugging Face: [nq\_open](https://huggingface.co/datasets/google-research-datasets/nq_open)
* **Model**: Transformer architecture with Multi-Head Attention, Feed Forward networks, and Linear layers.
* **Encoding**: Token and Positional Encoding
* **Tokenization**: Two-layer encoding process using tiktoken with subsequent compression.
* **Data Split**: 90% training, 10% testing (holdout method)
* **Batching**: Implemented dataset batches to optimize training, minimize overfitting, and reduce memory usage.
* **Model Persistence**: Trained model parameters saved using PyTorch's `torch.save()` and are accessible [here](https://github.com/jsmnlao/Google-2B-Search-Query-Recommendation-System/blob/main/saved_bigram_language_model.pth).
* **Performance Measurement**: Log loss
* **Training Specs**: Due to resource constraints, training was conducted on a CPU for 25 hours, with 10,000 iterations over 4,000 training records.
* **Evaluation**: Final input/output evaluations are available in [Final\_Search\_Query\_Model.ipynb](https://github.com/jsmnlao/Google-2B-Search-Query-Recommendation-System/blob/main/Final_Search_Query_Model.ipynb)

## Quick Start: Running the Training Code

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/jsmnlao/Google-2B-Search-Query-Recommendation-System.git](https://github.com/jsmnlao/Google-2B-Search-Query-Recommendation-System.git)
    ```
    OR
    ```bash
    git pull
    ```
2.  **Install Dependencies**:
    ```bash
    pip3 install -r requirements.txt
    ```
3.  **Execute Training**:
    ```bash
    python model.py
    ```

## Future Enhancements

We aim to refine our model by incorporating personalized queries, capturing long-term dependencies (using LSTMs), and recognizing patterns in common search phrases. We also plan to include key metrics like precision and recall, train with larger datasets on GPUs, and conduct extensive hyperparameter tuning for optimal performance.