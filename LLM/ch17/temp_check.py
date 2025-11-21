import torch.nn as nn
import numpy as np

# SimpleNERModel 클래스 정의: Named Entity Recognition(개체명 인식)을 위한 간단한 모델
# nn.Module을 상속받아 PyTorch 모델로 작동합니다.
class SimpleNERModel(nn.Module):
  # 모델 초기화 메서드
  # num_labels: 개체명 라벨(클래스)의 총 개수
  def __init__(self, num_labels) -> None:
    # 부모 클래스인 nn.Module의 생성자를 호출하여 초기화합니다.
    super(SimpleNERModel, self).__init__()
    # 라벨의 개수를 인스턴스 변수로 저장합니다.
    self.num_labels = num_labels
    # 사전 학습된 BERT 모델을 로드합니다. (MODEL_NAME은 외부에서 정의되어야 합니다.)
    # 이 BERT 모델은 입력 토큰을 임베딩하고 문맥 정보를 학습합니다.
    self.bert = AutoModel.from_pretrained(MODEL_NAME)
    # 드롭아웃 레이어를 정의합니다. 과적합을 방지하기 위해 0.1의 확률로 뉴런을 비활성화합니다.
    self.dropout = nn.Dropout(0.1)
    # 분류를 위한 선형 레이어(Fully Connected Layer)를 정의합니다.
    # BERT 모델의 출력 은닉 상태 크기를 입력으로 받고, 라벨 개수를 출력으로 가집니다.
    self.clf = nn.Linear(self.bert.config.hidden_size,  self.num_labels)

  # 모델의 순전파(forward pass)를 정의하는 메서드
  # input_ids: 입력 토큰 ID 시퀀스
  # attention_mask: 어텐션 마스크 (패딩 토큰을 무시하도록 합니다)
  def forward(self, input_ids, attention_maks):
    # BERT 모델을 사용하여 입력 시퀀스를 인코딩합니다.
    # outputs에는 last_hidden_state, pooler_output 등 다양한 정보가 포함됩니다.
    outputs = self.bert(input_ids, attention_mask=attention_maks)
    # BERT의 마지막 은닉 상태(hidden state)를 추출합니다.
    # 이 상태는 각 입력 토큰에 대한 문맥적 임베딩을 포함합니다.
    sequence_output =  outputs.last_hidden_state
    # 추출된 은닉 상태에 드롭아웃을 적용합니다.
    sequence_output = self.dropout(sequence_output)
    # 드롭아웃이 적용된 은닉 상태를 분류기(선형 레이어)에 통과시켜 로짓(logit)을 계산합니다.
    # 로짓은 각 라벨에 대한 분류 점수를 나타냅니다.
    logits = self.clf(sequence_output)
    # 계산된 로짓을 반환합니다.
    return logits
