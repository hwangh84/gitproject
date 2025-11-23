# LLM2 폴더 요약 및 모델 비교

이 문서는 `LLM2` 폴더 내의 파일들을 분석하여 주요 학습 내용과 사용된 AI 모델들을 요약하고 비교한 자료입니다.

## 📂 폴더 개요

`LLM2` 폴더는 자연어 처리(NLP)와 대형 언어 모델(LLM)을 중심으로 한 교육 자료 및 실습 코드를 포함하고 있습니다. 기초적인 NLP 이론부터 최신 LLM 활용 기술까지 폭넓은 주제를 다루고 있습니다.

### 주요 학습 모듈
1.  **NLP 기초**: 자연어 처리 개요, 전처리, 토큰화 (WordPiece, BPE).
2.  **Language Model 발전사**: 통계적 모델(N-gram) → 딥러닝(RNN/LSTM) → 트랜스포머(Attention, BERT/GPT).
3.  **LLM 실전 활용**: Hugging Face 라이브러리, 프롬프트 엔지니어링, RAG(검색 증강 생성), 에이전트(LangChain, LangGraph).
4.  **LLM 고급 튜닝**: 파인튜닝(Fine-tuning), PEFT(LoRA, Adapter).
5.  **컴퓨터 비전 및 멀티모달**: CNN, YOLO(객체 탐지), GAN/Diffusion(이미지 생성), CLIP.

---

## 📊 모델 비교표

폴더 내에서 확인된 주요 모델들의 특징과 용도를 비교한 표입니다.

| 모델명 | 유형 | 주요 아키텍처 | 주요 용도 | 특징 | 관련 챕터 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BERT** | Encoder-only | Transformer | 텍스트 분류, 질의응답(QA), 개체명 인식 | 문맥을 양방향으로 이해하는 데 탁월함. | Ch15, Ch16, Ch19 |
| **DistilBERT** | Encoder-only | Transformer | 텍스트 분류, QA | BERT의 경량화 버전. 속도가 빠르고 메모리 효율적. | Ch19 |
| **KoBERT** | Encoder-only | Transformer | 한국어 텍스트 분류 | SKT에서 개발한 한국어 특화 BERT 모델. | Ch17 |
| **KoELECTRA** | Encoder-only | Transformer (ELECTRA) | 한국어 질의응답(QA) | Generator-Discriminator 구조로 학습 효율성 및 성능 우수. | Ch19 |
| **T5** | Encoder-Decoder | Transformer | 문서 요약, 번역 | 모든 NLP 태스크를 "Text-to-Text" 문제로 변환하여 해결. | Ch18 |
| **BART** | Encoder-Decoder | Transformer | 문서 요약, 생성 | BERT(인코더)와 GPT(디코더)의 장점을 결합. | Ch18 |
| **KoBART** | Encoder-Decoder | Transformer | 한국어 요약, 생성 | SKT에서 개발한 한국어 특화 BART 모델. | Ch18 |
| **YOLOv8** | Vision | CNN 기반 | 실시간 객체 탐지 | 빠른 속도와 높은 정확도로 실시간 영상 처리에 적합. | rpi_yolo... |
| **CNN** | Vision | Convolutional NN | 이미지 분류, 텍스트 분류 | 이미지의 패턴 인식에 강점. 텍스트 분류에도 활용 가능. | Ch12, Ch18 |

---

## 📝 주요 개념 요약

### 1. Attention Mechanism (Ch13)
- **핵심**: Seq2Seq 모델의 정보 손실 문제를 해결하기 위해 등장. "필요할 때, 필요한 부분만 집중"하는 메커니즘.
- **구성**: Query(질문), Key(목차), Value(내용).
- **Self-Attention**: 문장 내 단어들이 서로의 관계를 파악하여 문맥을 이해하는 트랜스포머의 핵심 기술.

### 2. 문서 요약 (Ch18)
- **추출적 요약 (Extractive)**: 원문에서 중요 문장을 그대로 발췌.
- **생성적 요약 (Abstractive)**: 원문의 의미를 이해하고 새로운 문장을 생성 (T5, BART 등 사용).

### 3. 질의 응답 (QA) (Ch19)
- **Extractive QA**: 지문(Context) 내에서 정답의 시작과 끝 위치를 찾아 추출 (BERT 등 사용).
- **Generative QA**: 질문을 이해하고 답변을 직접 생성.

### 4. RAG (Retrieval-Augmented Generation)
- LLM이 학습하지 않은 외부 지식을 검색(Retrieval)하여 답변 생성(Generation)에 활용하는 기술. 환각(Hallucination) 현상을 줄이고 최신 정보를 반영할 수 있음.

### 5. 라즈베리파이 객체 인식 (YOLO)
- 라즈베리파이와 윈도우 PC를 연동하여 YOLOv8 모델로 실시간 객체 인식을 수행하는 가이드 포함.
