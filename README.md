# DACON-CodeSimilarity - 코드 유사성 판단 AI 경진대회

## Pre-trained CodeBERTa, GraphCodeBERT, electra, MiniLM Ensemble

+ 주최 및 주관: 데이콘 
+ 링크: https://dacon.io/competitions/official/235900/overview/description

+ 두 코드간 유사성(동일 결과물 산출 가능한지) 여부를 판단하는 모델 개발 

<img width="994" alt="image" src="https://user-images.githubusercontent.com/30611947/192323039-d40a1e12-0d12-4572-8912-6def5bb6711d.png">

----
## Summary
+ ### Data Processing
    + 간단한 전처리 수행 
      + '주석 -> 삭제'
      + '    ' -> tab 변환
      + 다중 개행 -> 한 번으로 변환
      + 일정 길이가 넘는 token은 truncation으로 인해 버려지기 때문에 일정 길이 이상이면 제외


    + [code1, code2] pair가 훈련에 사용에 됐는데, [code1, code3]이 validation 학습에 들어가면 문제가 생길 여지가 존재 -> pair 구성 전 train, validation 분리

    + BM25 알고리즘을 사용하여 최대한 비슷한 코드를 선택 후 높은 순서로 정렬 후 다른 문제를 푸는 코드들만 negative pair로 구성

    + Parameter copy를 통해, Pre-trained model의 레이어를 깊게 쌓음

    + 학습 시간이 길기 때문에 학습 속도 개선을 위해 Uniform Length Batch 시도


----
  
+ ### Model
    + Pre-trained electra, MiniLM, CodeBERTa, GraphCodeBERT 모델들을 사용함 
   
      + 마이크로소프트의 (Graph)CodeBert 모델의 경우, 파이썬, 자바스크립트 등의 소스 코드(사전 학습 데이터가 파인 튜닝에 사용되는 데이터와 비슷함)를 바탕으로 토크나이저가 만들어지고 또 사전 학습된 상태라서 토크나이징이 짧게 잘 이뤄질 수 있음
    
      + 레이어 복제(Parameter copy)를 통해 Pre-trained model의 레이어를 늘림 
      
      + Soft voting ensemble 수행 
  
  </br>

    + Model techniques
      + Dataloder : 속도 향상을 위한 UniformLengthBatchingSampler 사용
      + scheduler: CosineAnnealingLR
      + Loss : SmoothCrossEntropyLoss 사용
      + optimizer : AdamW 사용
      + EarlyStopping 사용
      + automatic mixed precision 사용

----

+ ## Environment 
  + 사용한 Docker image는 Docker Hub에 첨부하며 cuda10.2, cudnn7, ubuntu18.04 환경을 제공합니다.
    + https://hub.docker.com/r/lsy2026/code_similarity
  
  
+ ## Libraries
  + python==3.9.7
  + pandas==1.3.4
  + numpy==1.20.3
  + tqdm==4.62.3
  + sklearn==0.24.2
  + torch==1.10.2+cu102
  + transformers==4.19.1
  + re==2.2.1
  + rank_bm25==0.2.2

---- 

+ ## 개선할 점
  
  + contrastive learning 시도하여 다양한 방식의 접근 방식 고려
  + 파인튜닝 데이터로 MLM 모델의 vocab에 데이터 추가하여 MLM 학습 후 파인튜닝 하는 방법(TAPT) 방식 고려
  + Pre-trained MLM과 유사한 방식으로 문제를 풀어나가게 변경 -> classification 문제를 cloze task로 변경하는 방식(Pattern Exploiting Training, PET) 고려
  + 여러 언어로 학습 후 여러 언어로 inference 진행하여 ensemble(Multi-lingual) -> 해당 방법은 Back-translation 처럼 다른 NLP task에 적용 가능할 것 같아보임
  + 토크나이징 시 512개를 넘어가서 truncation 하는 경우 왼쪽에서 잘라내기 -> 많은 코드들이 라이브러리 임포트 시 주석을 다는 경우가 많음
