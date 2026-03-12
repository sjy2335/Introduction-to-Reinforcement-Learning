import numpy as np


def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    """
    주어진 정책에 대한 가치 함수 V 계산
    V(s) = Σ_a π(a|s) * Σ_{s'} P(s'|s,a) * [R + gamma * V(s')]
    """
    V = np.zeros(env.nS)

    while True:
        delta = 0
        for s in range(env.nS):
            v = 0
            for a in range(env.nA):
                for prob, next_state, reward in env.MDP[s][a]:
                    v += policy[s][a] * prob * (reward + gamma * V[int(next_state)])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break

    return V


def policy_improvement(env, V, gamma=0.99):
    """
    가치 함수 V 기반으로 그리디하게 더 나은 정책 도출
    """
    policy = np.zeros([env.nS, env.nA]) #/ env.nA

    for s in range(env.nS):
        q_values = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward in env.MDP[s][a]:
                q_values[a] += prob * (reward + gamma * V[int(next_state)])
        best_action = np.argmax(q_values) # 최대값을 가지는 action 중 가장 작은 index 선택
        policy[s][best_action] = 1.0
        
    return policy


def policy_iteration(env, gamma=0.99, theta=1e-8):
    """
    정책 평가, 정책 개선, 정책 수렴 확인 반복
    """
    V = np.zeros(env.nS)
    policy = np.ones([env.nS, env.nA]) / env.nA  # 초기 정책: 모든 행동 균등 확률

    while True:
        # 1) 정책 평가: 현재 정책 π에 대해 V를 수렴할 때까지 계산
        V = policy_evaluation(env, policy, gamma, theta)
        
        # 2) 정책 개선: V를 기반으로 각 상태에서 최적 행동을 선택한 새 정책 생성
        new_policy = policy_improvement(env, V, gamma)

        # 3) 수렴 확인: 정책이 더 이상 바뀌지 않으면 최적 정책에 도달
        if np.array_equal(new_policy, policy):
            break
        
        policy = new_policy

    return policy, V


def value_iteration(env, gamma=0.99, theta=1e-8):
    """
    가치 함수를 직접 반복적으로 갱신 후 정책 추출
    V(s) = max_a Σ_{s'} P(s'|s,a) * [R + gamma * V(s')]
    """
    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA])

    # 1) 가치 갱신: 정책 평가 없이, 각 상태에서 max Q값을 바로 V에 반영
    while True:
        delta = 0
        for s in range(env.nS):
            q_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward in env.MDP[s][a]:
                    q_values[a] += prob * (reward + gamma * V[int(next_state)])
            best_value = np.max(q_values)  # policy_iteration과 달리 max를 바로 취함
            delta = max(delta, abs(best_value - V[s]))
            V[s] = best_value
        if delta < theta:
            break

    # 2) 정책 추출: 수렴된 V로부터 마지막에 한 번만 최적 정책을 결정
    for s in range(env.nS):
        q_values = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward in env.MDP[s][a]:
                q_values[a] += prob * (reward + gamma * V[int(next_state)])
        best_action = np.argmax(q_values)
        policy[s][best_action] = 1.0

    return policy, V