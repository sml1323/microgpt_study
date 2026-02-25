# 🧠 MicroGPT 디버그 학습 가이드

> Karpathy의 microgpt를 디버거로 한 줄씩 따라가며 GPT의 본질을 이해하기 위한 학습 자료입니다.

## 📁 파일 구성

| 파일                           | 목적                                 | 소요 시간 |
| ------------------------------ | ------------------------------------ | --------- |
| `01_autograd_playground.py`    | Value 클래스만 떼어서 자동 미분 이해 | 30분      |
| `02_microgpt_debug.py`         | 축소 GPT 전체를 디버거로 추적        | 1~2시간   |
| `03_overfitting_experiment.py` | 단어 1개 오버피팅으로 학습 원리 체감 | 30분      |

## 🚀 시작하기

### 1. VS Code 디버그 실행

1. VS Code 좌측 사이드바에서 **벌레 아이콘** (Run and Debug) 클릭
2. 상단 드롭다운에서 단계 선택 (예: `1단계: Autograd 실험`)
3. **F5** 누르면 디버그 모드로 실행!

### 2. 필수 Watch 표현식

Value 객체는 디버거에서 `<Value object at 0x...>`로 보여서 답답합니다.
**Watch 탭**에 아래 표현식들을 등록해두세요:

```python
# 벡터 내부 값 보기
[v.data for v in x]

# 확률 분포 보기
[round(p.data, 4) for p in probs]

# 로짓 (raw 점수) 보기
[round(v.data, 4) for v in logits]

# 어텐션 가중치 보기
[round(w.data, 4) for w in attn_weights]

# 그래디언트 보기
[round(p.grad, 6) for p in params[:8]]
```

---

## 📖 4단계 학습 순서

### 1단계: 자동 미분의 마법 (01_autograd_playground.py)

**목표**: `backward()`가 어떻게 그래디언트를 계산하는지 이해

**핵심 질문**:

- [ ] `c = a * b + a`에서 `a.grad`가 왜 4인가? (a가 두 곳에서 쓰이니까!)
- [ ] `backward()` 안의 `topo` 리스트에 노드들이 어떤 순서로 담기는가?
- [ ] `child.grad += local_grad * v.grad`에서 왜 `=`가 아니라 `+=`인가?
- [ ] 실험 3의 loss.backward() 후, 정답 로짓의 grad는 왜 음수인가?

**디버깅 방법**:

1. `c.backward()`에 브레이크포인트 → **F11 (Step Into)**
2. `build_topo` 재귀 호출을 따라가며 `topo` 리스트 관찰
3. `reversed(topo)` 루프에서 `child.grad`가 채워지는 과정 확인

---

### 2단계: GPT의 심장부 (02_microgpt_debug.py)

**목표**: 토큰 하나가 모델을 통과하며 "다음 토큰 예측"으로 변환되는 과정 추적

**핵심 질문**:

- [ ] `tok_emb + pos_emb`는 무엇을 의미하는가? (무엇 + 어디)
- [ ] 어텐션 가중치가 의미하는 것은? (현재 → 과거의 어떤 글자에 집중?)
- [ ] `softmax(logits)` 결과에서 가장 높은 확률을 가진 토큰은?
- [ ] 학습 전후로 `probs[target_id].data`가 어떻게 변하는가?

**디버깅 시나리오**: "emma" 학습 추적

```
토큰: [BOS, e, m, m, a, BOS] (실제 인덱스: [26, 4, 12, 12, 0, 26])

pos=0: BOS → 모델 → probs → 정답은 'e' → loss 계산
pos=1: e   → 모델 → probs → 정답은 'm' → loss 계산
pos=2: m   → 모델 → probs → 정답은 'm' → loss 계산
pos=3: m   → 모델 → probs → 정답은 'a' → loss 계산
```

---

### 3단계: 학습의 본질 — 오버피팅 (03_overfitting_experiment.py)

**목표**: 모델이 데이터의 패턴을 파라미터에 저장하는 과정 관찰

**핵심 질문**:

- [ ] 20스텝마다 생성되는 결과가 점점 "karpathy"에 가까워지는가?
- [ ] loss가 1.0 이하로 떨어지면 생성 품질에 어떤 변화가?
- [ ] `TARGET_WORD`를 "hello"로 바꾸면 결과가 어떻게 달라지는가?

---

### 4단계: 원본 실행 (microgpt.py)

모든 단계를 마친 후, 원본 `microgpt.py`를 한 번 돌려보세요.
32,000개 이름으로 학습하면 어떤 새로운 이름이 나오는지 관찰!

```bash
python microgpt.py  # 약 1분 소요
```

---

## 🔑 핵심 개념 치트시트

| 개념              | 한 줄 설명                                                   |
| ----------------- | ------------------------------------------------------------ |
| **Value**         | 숫자 하나 + 미분값을 감싼 래퍼. PyTorch의 Tensor와 같은 역할 |
| **Forward**       | 입력 → 계산 → 출력. 연산 그래프를 만드는 과정                |
| **Backward**      | 출력(loss)에서 입력(params)으로 역방향 미분. 체인룰 적용     |
| **Embedding**     | 정수 ID → 벡터. 룩업 테이블에서 행 하나를 가져오는 것        |
| **Attention**     | "이 글자가 다음을 예측할 때 과거의 어떤 글자를 참고할까?"    |
| **MLP**           | 위치별 독립 계산. 어텐션이 모은 정보를 "사고"하는 부분       |
| **Residual**      | `x = f(x) + x`. 그래디언트가 깊은 네트워크를 잘 흐르게 함    |
| **Softmax**       | 로짓 → 확률. 합이 1이 되고 모든 값이 0~1                     |
| **Cross-entropy** | `-log(정답의 확률)`. 정답 확률이 높으면 loss↓                |
| **Adam**          | 똑똑한 경사하강법. 모멘텀 + 적응적 학습률                    |
