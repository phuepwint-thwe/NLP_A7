# NLP_A7 : Training Distillation vs LoRA

- **Name:** Phue Pwint Thwe  
- **ID:** st124748

This project focuses on building a hate speech classification model using techniques like **Distillation** and **LoRA (Low-Rank Adaptation)**. A web application is also developed using **Streamlit** to make predictions in real-time.

---

## Task Overview

### Task 1: Dataset Preprocessing
- Loaded **HateXplain dataset** from Hugging Face (`hate_speech_offensive`).
- Extracted relevant columns (`tweet`, `class`) and mapped:
  - `0`: Non-Hate
  - `1`: Offensive
  - `2`: Hate
- Tokenized text using `bert-base-uncased` tokenizer.
- Prepared PyTorch-ready datasets using dynamic padding.

---

### Task 2: Odd vs Even Layer Distillation
- Used a **12-layer BERT model** as the teacher.
- Built two **6-layer student models**:
  - **Odd-Layer Student:** Used layers `{1, 3, 5, 7, 9, 11}`.
  - **Even-Layer Student:** Used layers `{2, 4, 6, 8, 10, 12}`.
- Trained both students using a mix of:
  - Classification Loss (CE)
  - KL Divergence Loss
  - Cosine Embedding Loss
- **Result:**
  - Odd Student Accuracy: `98.2%`
  - Even Student Accuracy: `95.9%`

---

### Task 3: LoRA Fine-Tuning
- Applied **LoRA** to the **full 12-layer student model**.
- Only trained adapter weights for attention modules (`query`, `value`).
- Trained for 5 epochs with minimal compute cost.
- **Result:** LoRA Student Accuracy: `89.8%`

---

### Task 4: Evaluation & Analysis
| Model         | Final Train Loss | Test Accuracy |
|---------------|------------------|---------------|
| Odd-Layer     | 0.2731           | **98.2%**     |
| Even-Layer    | 0.2875           | 95.9%         |
| LoRA (Full)   | 0.3563           | 89.8%         |

- Odd-Layer Student is the best performing model.
- LoRA improves over baseline but struggles to match distilled models.
- Fine-tuning LoRA further (higher rank r=16 or longer training) may help.
- Even-Layer model improved steadily but remains below Odd-Layer.

---

### Task 5: Streamlit Web App
- Built an intuitive **web application** using **Streamlit**.
- User inputs a sentence and gets classification result (Non-Hate / Offensive / Hate).
- Used `student_model_odd` as the backend model.
- Cached model loading for fast response.

#### Example Outputs


#### How to Run:
```bash
streamlit run app.py
