# 🔍 Google AI Studio - LLM Search Query Recommendation 🚀

<div align="center">
  <img src="https://media1.tenor.com/m/SWJNyLKohhkAAAAC/google-gif.gif" width="500" alt="Cool Google GIF">
</div>

**Purpose**: This project 💡 was designed to cultivate a deep understanding of the data and technical skills 💻 necessary to build a robust search engine 🌐, similar to Google’s. Our primary objective was to develop an LLM capable of replicating the predictive autocomplete functionality ⌨️ found in modern search engines, built entirely from scratch.

Explore our presentation slides [here](https://docs.google.com/presentation/d/1h4i1ClAScYdBZICnKjBKxB94AW4P4HrFIiRsZ9Y-uOQ/edit?usp=sharing)! 📚

## Meet the Team 🧑‍🤝‍🧑

**Team Members**: Viswaretas Kotra, Rexford Nimoh

**Challenge Advisors & TAs**: Josh McAdams, Esther Lou

## Tech Stack 🛠️

* Hugging Face 🤗
* Google Colab 🧪
* Pandas 🐼
* PyTorch 🔥
* TikToken 🪙
* Visual Studio Code 💻

## Methodology & Implementation ⚙️

* **Dataset**: Sourced from Hugging Face: [nq\_open](https://huggingface.co/datasets/google-research-datasets/nq_open) 📊
* **Model**: Transformer architecture with Multi-Head Attention, Feed Forward networks, and Linear layers. 🧠
* **Encoding**: Token and Positional Encoding ✍️
* **Tokenization**: Two-layer encoding process using tiktoken with subsequent compression. 压缩
* **Data Split**: 90% training, 10% testing (holdout method) ➗
* **Batching**: Implemented dataset batches to optimize training, minimize overfitting, and reduce memory usage. 📦
* **Model Persistence**: Trained model parameters saved using PyTorch's `torch.save()` and are accessible [here](https://github.com/jsmnlao/Google-2B-Search-Query-Recommendation-System/blob/main/saved_bigram_language_model.pth). 💾
* **Performance Measurement**: Log loss 📉
* **Training Specs**: Due to resource constraints, training was conducted on a CPU for 25 hours, with 10,000 iterations over 4,000 training records. ⏱️
* **Evaluation**: Final input/output evaluations are available in [Final\_Search\_Query\_Model.ipynb](https://github.com/jsmnlao/Google-2B-Search-Query-Recommendation-System/blob/main/Final_Search_Query_Model.ipynb) 📝

## Quick Start: Running the Training Code ▶️

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

## Future Enhancements ✨

We aim to refine our model by incorporating personalized queries, capturing long-term dependencies (using LSTMs), and recognizing patterns in common search phrases. We also plan to include key metrics like precision and recall, train with larger datasets on GPUs, and conduct extensive hyperparameter tuning for optimal performance. 🚀
