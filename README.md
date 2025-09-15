# 🛠️ AI Terms of Service Analyzer

## 📌 Overview
This project was developed as part of the AGI Agent Application Hackathon. It aims to solve the problem of understanding complex legal terms in various service agreements. Using RAG (Retrieval-Augmented Generation) technology and AI agents, the system automatically analyzes terms of service documents, identifies relevant legal provisions, and provides user-friendly legal advice.

## 🚀 Key Features
✅ **Domain Classification**: Automatically classifies uploaded terms of service into categories (housing lease, subscription service, or other)
✅ **Legal Provision Retrieval**: Uses RAG to find relevant legal provisions for each clause in the terms
✅ **AI-Powered Analysis**: Provides user-friendly explanations of legal implications
✅ **Automated Email Inquiries**: Generates and sends automated inquiry emails based on analysis results

## 🖼️ Demo
![demo](./assets/lawviser_demo.gif)

## 🧩 Tech Stack
- **Frontend**: Streamlit
- **Backend**: Python
- **Database**: FAISS vector database
- **Others**: SentenceTransformer, PyTorch, HuggingFace Transformers, LoRA fine-tuning

## 🏗️ Project Structure
```
AGI-Hackathon/
├── app.py                  # Streamlit web application (main UI)
├── main.py                 # Command-line interface (CLI)
├── requirements.txt        # Project dependencies
│
├── RAG/                    # RAG (Retrieval-Augmented Generation) related code
│   ├── rag.py              # Core RAG system logic
│   ├── embedding.py        # Embedding processing module
│   ├── model.py            # Model handler
│   ├── generation.py       # Answer generation module
│   ├── build_db.py         # Vector database construction
│   ├── evaluation.py       # Evaluation module
│   ├── housing_db/         # Housing lease vector DB
│   ├── subscribe_db/       # Subscription service vector DB
│   ├── other_db/           # Other domain vector DB
│   └── legal_data/         # Legal data repository
│
├── tools/                  # Document classification and processing tools
│   ├── classifier.py       # Document classifier (lease/subscription/other)
│   ├── classifier_decoder.py # Classification result decoder (not use)
│   ├── detector.py         # Document feature detector (not use) 
│   └── email_module.py     # Email automation functionality
│
├── lm/                     # Language model related code
│   ├── final_answer.py     # Final answer generation module
│   ├── install.sh          # install tmux
│   ├── requirements.txt    # finetuning dependencies
│   ├── run.sh              # run tmux
│   ├── train_eval.py       # Model training and evaluation
│   ├── lora_weights/       # LoRA weights repository
│   ├── configs/            # configs
│   ├── utils/              # utils
│   └── n_shot_prompts/     # Prompt templates
│
├── datasets/               # Datasets
│   ├── judgedata/          # Legal case data
│   └── termdata/           # Terms of service data
│
├── examples/               # example Datasets
│
└── results/                # Results storage directory
```

## 🔧 Setup & Installation
```bash
# Clone the repository
git clone https://github.com/MaDoKaLiF/LawViser.git

# Install dependencies
pip install -r requirements.txt

# Run the web interface
streamlit run app.py

# Alternatively, run the CLI version
python main.py
```

## 📁 Dataset & References
**Datasets used:**
- Legal case data: Korean legal precedents and statutes related to housing leases and subscription services
- Terms of service examples: Collection of various service agreements for training and testing

**References / Resources:**
- Korean Consumer Protection Act
- Housing Lease Protection Act
- E-commerce Transaction Protection Act

## 🙌 Team Members
| Name | Role | GitHub |
|------|------|--------|
| 양인혜 | RAG System Developer | @1nhye |
| 김형진 | Document Classification Developer | @MaDoKaLiF |
| 이준찬 | DataBase Administrator | @gamma4638 |
| 장재인 | Language Model Engineer | @Jaein-Jang |
| 최서영 | Project Manager | @choi613504 |

## ⏰ Development Period
Last updated: 2025-04-05

## 📄 License
This project is licensed under the MIT license.
See the LICENSE file for more details.

## 💬 Additional Notes
The system currently supports Korean language terms of service documents and provides analysis based on Korean legal framework. Future work may include support for additional languages and legal jurisdictions.
