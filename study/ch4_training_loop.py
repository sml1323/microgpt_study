"""
=============================================================================
 Chapter 4: Training Loop (학습 루프)
 - microgpt.py L145~183에 해당하는 코드
 - explain.md의 Training loop, Adam optimizer 섹션 내용을 한국어 주석으로 포함
 - "emma" 단어 하나로 5스텝만 학습 (오버피팅 실험)
 - 독립 실행 가능: python study/ch4_training_loop.py
=============================================================================
"""

import os
import math
import random
random.seed(42)

# 디버거(LazyVim 등)에서 cwd가 study/일 때 input.txt를 찾기 위해 프로젝트 루트로 이동
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# ─────────────────────────────────────────────────────────────────────────────
# 사전 준비: Value 클래스
# ─────────────────────────────────────────────────────────────────────────────
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# ─────────────────────────────────────────────────────────────────────────────
# 사전 준비: 토크나이저
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)

uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1

# ─────────────────────────────────────────────────────────────────────────────
# 사전 준비: 모델 (축소 파라미터)
# ─────────────────────────────────────────────────────────────────────────────
n_layer = 1
n_embd = 4        # 축소: 원본 16 → 4
block_size = 8    # 축소: 원본 16 → 8
n_head = 2        # 축소: 원본 4 → 2
head_dim = n_embd // n_head

matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = [p for mat in state_dict.values() for row in mat for p in row]

def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)
    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]
    logits = linear(x, state_dict['lm_head'])
    return logits

print(f"vocab_size: {vocab_size}, params: {len(params)}")


# =============================================================================
# 섹션 1: 학습 루프 해부 (microgpt.py L145~183)
# =============================================================================
#
# [explain.md - Training loop 섹션 요약]
# 학습 루프는 반복적으로 다음을 수행한다:
#   (1) 문서를 선택한다
#   (2) 모델을 순전파(forward)시킨다
#   (3) 손실(loss)을 계산한다
#   (4) 역전파(backward)로 그래디언트를 구한다
#   (5) 파라미터를 업데이트한다
#
# ──────── 학습 루프의 각 단계 ────────
#
# 토큰화 (Tokenization):
#   각 학습 스텝은 하나의 문서를 선택하고 양쪽에 BOS를 붙인다:
#   "emma" → [BOS, e, m, m, a, BOS]
#   모델의 임무: 이전 토큰들이 주어졌을 때 각각의 다음 토큰을 예측하는 것.
#
# 순전파와 손실 (Forward pass and loss):
#   토큰들을 하나씩 모델에 넣어 KV 캐시를 쌓아간다.
#   각 위치에서 모델은 27개 로짓 출력 → softmax로 확률 변환.
#   손실 = 정답 토큰의 음의 로그 확률: -log(p(target))
#   이것이 cross-entropy loss.
#   직관: 모델이 실제 다음 토큰에 얼마나 "놀랐는가"
#   - 정답에 확률 1.0 → 손실 0 (전혀 놀라지 않음)
#   - 정답에 확률 ~0   → 손실 → +∞ (매우 놀람)
#
# 역전파 (Backward pass):
#   loss.backward() 한 번 호출로 전체 계산 그래프를 역전파.
#   각 파라미터의 .grad에 "이 파라미터를 어떻게 바꾸면 loss가 줄어드는지" 저장.
#
# [explain.md - Adam optimizer 섹션 요약]
# Adam 옵티마이저:
#   단순히 p.data -= lr * p.grad (경사 하강법)를 사용할 수도 있지만,
#   Adam은 더 똑똑하다:
#   - m: 최근 그래디언트의 평균 (모멘텀, 굴러가는 공처럼)
#   - v: 최근 그래디언트 제곱의 평균 (파라미터별 학습률 조절)
#   - m_hat, v_hat: 편향 보정 (m, v가 0으로 초기화되어 워밍업 필요)
#   - 학습률은 학습 중 선형으로 감소
#   - 업데이트 후 .grad = 0으로 리셋 (다음 스텝을 위해)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("🏋️ [학습 실험] 'emma' 하나로 5스텝 학습")
print("=" * 60)

# --- 원본 코드 (microgpt.py L145~148) ---
# Adam 옵티마이저 초기화
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m_buf = [0.0] * len(params)  # 1차 모멘텀 버퍼 (그래디언트 평균)
v_buf = [0.0] * len(params)  # 2차 모멘텀 버퍼 (그래디언트 제곱 평균)

# 학습할 단어
doc = "emma"
tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
n = len(tokens) - 1  # 예측할 위치 수 = 5, block_size보다 작으므로 그대로

num_steps = 5
print(f"\n  학습 단어: '{doc}'")
print(f"  토큰열: {tokens} (= [BOS, 'e', 'm', 'm', 'a', BOS])")
print(f"  예측 위치 수: {n}")
print(f"  학습 스텝 수: {num_steps}")

# 학습 전 파라미터 몇 개 스냅샷 (비교용)
sample_params_idx = [0, 1, 2]
print(f"\n  [학습 전] 파라미터 샘플 (wte[0][:3]):")
for i in sample_params_idx:
    print(f"    params[{i}]: data={params[i].data:+.6f}")

print(f"\n{'─' * 60}")

# =============================================================================
# 학습 루프 (microgpt.py L152~183)
# =============================================================================
for step in range(num_steps):
    print(f"\n  ══════ STEP {step + 1}/{num_steps} ══════")

    # ─── (1) Forward pass: 토큰을 모델에 넣기 ──────────────────
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []

    print(f"\n  [Forward Pass]")
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)

        # ─── 디버그 포인트: 각 위치의 예측 관찰 ─────────────────
        input_str = "[BOS]" if token_id == BOS else f"'{uchars[token_id]}'"
        target_str = "[BOS]" if target_id == BOS else f"'{uchars[target_id]}'"
        target_prob = probs[target_id].data

        # Top 3 예측
        prob_pairs = sorted([(i, p.data) for i, p in enumerate(probs)], key=lambda x: -x[1])[:3]
        top3_str = ", ".join(
            f"{'[BOS]' if idx == BOS else uchars[idx]}:{p:.3f}" for idx, p in prob_pairs
        )

        print(f"    pos {pos_id}: {input_str:>5s} → 정답={target_str:>5s}  "
              f"P(정답)={target_prob:.4f}  loss={loss_t.data:.4f}  "
              f"top3=[{top3_str}]")

    # ─── (2) 평균 손실 계산 ──────────────────────────────────
    loss = (1 / n) * sum(losses)
    print(f"\n  [Loss] 평균 손실 = {loss.data:.4f}")

    # ─── (3) Backward pass ──────────────────────────────────
    loss.backward()

    # ─── 디버그 포인트: backward 후 그래디언트 관찰 ──────────
    print(f"\n  [Backward 후 — 그래디언트 샘플]")
    # wte 임베딩의 그래디언트를 관찰
    print(f"    wte[BOS={BOS}] (학습에 사용된 임베딩):")
    bos_emb = state_dict['wte'][BOS]
    for j in range(min(n_embd, 4)):
        print(f"      wte[{BOS}][{j}]: data={bos_emb[j].data:+.6f}, grad={bos_emb[j].grad:+.6f}")

    # 사용되지 않은 토큰의 그래디언트도 확인
    unused_token = 25  # 'z' — "emma"에 없는 문자
    unused_emb = state_dict['wte'][unused_token]
    unused_grad = sum(abs(unused_emb[j].grad) for j in range(n_embd))
    print(f"    wte['z'={unused_token}] grad 합 = {unused_grad:.6f}  ← 학습에 안 쓰여서 0에 가까움!")

    # ─── (4) Adam 옵티마이저 업데이트 ─────────────────────────
    #
    # [explain.md 요약]
    # m[i] = β₁ · m[i] + (1-β₁) · grad         ← 모멘텀 (그래디언트 이동 평균)
    # v[i] = β₂ · v[i] + (1-β₂) · grad²        ← 그래디언트 크기 추적
    # m_hat = m[i] / (1 - β₁^(t+1))             ← 편향 보정
    # v_hat = v[i] / (1 - β₂^(t+1))             ← 편향 보정
    # p.data -= lr_t · m_hat / (√v_hat + ε)      ← 파라미터 업데이트
    #
    # 직관: "그래디언트가 꾸준히 같은 방향이면 더 크게 움직이고 (m),
    #       그래디언트가 큰 파라미터는 학습률을 줄인다 (v)"

    lr_t = learning_rate * (1 - step / num_steps)  # 선형 학습률 감소

    # 업데이트 전후 비교를 위해 스냅샷
    before_data = [params[i].data for i in sample_params_idx]

    for i, p in enumerate(params):
        m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * p.grad
        v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad ** 2
        m_hat = m_buf[i] / (1 - beta1 ** (step + 1))
        v_hat = v_buf[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0  # 다음 스텝을 위해 그래디언트 리셋!

    # ─── 디버그 포인트: Adam 업데이트 전후 비교 ────────────────
    print(f"\n  [Adam 업데이트] lr_t = {lr_t:.4f}")
    print(f"    파라미터 변화 샘플 (wte[0][:3]):")
    for idx, i in enumerate(sample_params_idx):
        delta = params[i].data - before_data[idx]
        print(f"      params[{i}]: {before_data[idx]:+.6f} → {params[i].data:+.6f}  (Δ={delta:+.6f})")

print(f"\n{'─' * 60}")


# =============================================================================
# 학습 결과 관찰: loss가 줄어들었는가?
# =============================================================================

print("\n" + "=" * 60)
print("📉 [결과] 학습 후 — 같은 문서로 다시 forward")
print("=" * 60)

# 학습 후 같은 단어를 다시 넣어보기
keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
losses = []
print(f"\n  '{doc}' 재평가:")
for pos_id in range(n):
    token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
    logits = gpt(token_id, pos_id, keys, values)
    probs = softmax(logits)
    loss_t = -probs[target_id].log()
    losses.append(loss_t)

    input_str = "[BOS]" if token_id == BOS else f"'{uchars[token_id]}'"
    target_str = "[BOS]" if target_id == BOS else f"'{uchars[target_id]}'"
    target_prob = probs[target_id].data

    prob_pairs = sorted([(i, p.data) for i, p in enumerate(probs)], key=lambda x: -x[1])[:3]
    top3_str = ", ".join(
        f"{'[BOS]' if idx == BOS else uchars[idx]}:{p:.3f}" for idx, p in prob_pairs
    )

    print(f"    pos {pos_id}: {input_str:>5s} → 정답={target_str:>5s}  "
          f"P(정답)={target_prob:.4f}  loss={loss_t.data:.4f}  "
          f"top3=[{top3_str}]")

final_loss = (1 / n) * sum(l.data for l in losses)
print(f"\n  최종 평균 손실 = {final_loss:.4f}")
print(f"  → 학습 전 loss (~3.3) 대비 줄어들었는가? {'✅ Yes!' if final_loss < 3.3 else '아직 부족'}")


# =============================================================================
# 핵심 정리
# =============================================================================
print("\n" + "=" * 60)
print("🎯 핵심 정리")
print("=" * 60)
print(f"""
  학습 루프의 핵심 사이클:

  ┌─────────────────────────────────────────────┐
  │  1. Forward: 토큰 입력 → 모델 → 로짓 → 확률  │
  │  2. Loss: -log(정답 확률) = cross-entropy    │
  │  3. Backward: loss.backward() → 모든 grad    │
  │  4. Adam: grad 기반 파라미터 업데이트         │
  │  5. Reset: grad = 0, 다음 스텝 준비          │
  │  → 1로 돌아가서 반복!                        │
  └─────────────────────────────────────────────┘

  핵심 관찰:
  - loss가 스텝마다 (대체로) 줄어든다 → 모델이 학습하고 있다!
  - P(정답)이 점점 높아진다 → 예측이 점점 정확해진다
  - 사용되지 않은 토큰('z')의 grad ≈ 0 → 관련 없는 파라미터는 변하지 않는다
  - Adam의 모멘텀(m)과 적응적 학습률(v)이 단순 SGD보다 효과적

  💡 이 실험은 "emma" 하나에 오버피팅하는 것.
     실제 학습에서는 32,000개 이름을 순환하며 일반화를 학습한다.
     다음 단계: num_steps를 늘리거나, 여러 문서로 학습해보기!
""")
