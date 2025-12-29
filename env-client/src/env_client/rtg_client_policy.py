import time
import numpy as np
import threading
from collections import deque
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from env_client.websocket_client_policy import WebsocketClientPolicy

class RTGClientPolicy:
    def __init__(self, host, port, max_velocity=1.0, blending_horizon=10, alpha=0.5):
        self.base_policy = WebsocketClientPolicy(host, port)
        self.max_velocity = max_velocity
        self.blending_horizon = blending_horizon
        self.alpha = alpha # Decay rate for W1
        
        self.action_buffer = deque()
        self.executed_history = deque(maxlen=2) # For smoothness (x_{t-1}, x_{t-2})
        
        self.current_trajectory = None # Not strictly needed if we use buffer, but good for debugging
        
        self.inference_thread = None
        self.is_inferring = False
        self.inference_start_time = 0
        self.steps_since_inference_start = 0
        
        self.lock = threading.Lock()
        self.action_dim = None
        
    def reset(self):
        self.base_policy.reset()
        with self.lock:
            self.action_buffer.clear()
            self.executed_history.clear()
            self.is_inferring = False
            self.steps_since_inference_start = 0
        
    def infer_async(self, obs):
        with self.lock:
            if self.is_inferring:
                return
            self.is_inferring = True
            self.inference_start_time = time.time()
            self.steps_since_inference_start = 0
        
        def run_inference():
            try:
                res = self.base_policy.infer(obs)
                action_chunk = res["action"]
                self._process_new_chunk(action_chunk)
            except Exception as e:
                print(f"Inference failed: {e}")
            finally:
                with self.lock:
                    self.is_inferring = False

        self.inference_thread = threading.Thread(target=run_inference)
        self.inference_thread.start()

    def _process_new_chunk(self, action_chunk):
        # This runs in the thread
        A_new = np.array(action_chunk)
        if self.action_dim is None:
            self.action_dim = A_new.shape[1]
            
        with self.lock:
            # t1 + t2 is represented by steps_since_inference_start
            # (assuming 1 step per loop iteration in main thread)
            t_elapsed = self.steps_since_inference_start
            
            # If initial chunk (buffer empty or we just started)
            # Actually, if executed_history is empty, it's initial.
            is_initial = len(self.executed_history) == 0
            
            if is_initial:
                # Optimize over chunk itself
                optimized_chunk = self._optimize_trajectory(A_new, is_initial=True)
                self.action_buffer.extend(optimized_chunk)
            else:
                # RTG Logic
                # Discard A_new[0 : t_elapsed]
                if t_elapsed < len(A_new):
                    A_new_future = A_new[t_elapsed:]
                else:
                    # Inference took too long, new chunk is already old?
                    # Just append what's left or nothing
                    A_new_future = np.empty((0, self.action_dim))
                
                # A_old_future is what's currently in buffer
                # We need to convert deque to array
                A_old_future = np.array(self.action_buffer)
                
                # Blending
                # We blend for blending_horizon steps
                # Or min(len(A_new_future), len(A_old_future), blending_horizon)
                
                # Wait, we replace A_old with blended version?
                # "starting from t = t1 + t2, a smooth blending is performed between A_new and A_old"
                # "execution continues with A_old[0, t1 + t2]" -> This is what was already executed or is about to be.
                # The buffer currently holds A_old[t_elapsed_in_old : ]
                # Wait, steps_since_inference_start tracks how much we advanced since inference started.
                # So the current buffer IS A_old[t1+t2 : ].
                
                # So we blend current buffer with A_new_future.
                
                optimized_chunk = self._optimize_trajectory(A_new_future, A_old_future, is_initial=False)
                
                # Update buffer
                # The optimized chunk replaces the beginning of A_new_future AND A_old_future?
                # "The newly generated action chunk is represented as A_new... while the trajectory currently being executed is denoted by A_old"
                # "starting from t = t1 + t2, a smooth blending is performed... where tf is predefined blending horizon"
                # "with tf <= the last timestamp of A_old"
                
                # So we blend the overlapping part.
                # Then we append the rest of A_new.
                
                H = len(optimized_chunk)
                
                # Clear buffer and refill
                self.action_buffer.clear()
                self.action_buffer.extend(optimized_chunk)
                
                # Append rest of A_new if any
                if len(A_new_future) > H:
                    self.action_buffer.extend(A_new_future[H:])
                    
                # Spline interpolation?
                # "Finally, the optimized discrete trajectory is interpolated using splines"
                # This might be for upsampling or just smoothing. 
                # Given we are outputting discrete actions for the env step, maybe we just use the optimized points.
                # I'll skip explicit spline upsampling unless we need higher frequency control.

    def _optimize_trajectory(self, A_target, A_old_future=None, is_initial=False):
        H = len(A_target)
        if not is_initial and A_old_future is not None:
            H = min(H, len(A_old_future), self.blending_horizon)
        
        if H == 0:
            return A_target
            
        x0 = A_target[:H].flatten()
        
        prev1 = self.executed_history[-1] if len(self.executed_history) > 0 else np.zeros(self.action_dim)
        prev2 = self.executed_history[-2] if len(self.executed_history) > 1 else prev1
        
        def objective(x):
            x = x.reshape(H, -1)
            cost = 0
            
            # S1: Smoothness
            # t=0
            cost += np.sum((x[0] - 2*prev1 + prev2)**2)
            if H > 1:
                # t=1
                cost += np.sum((x[1] - 2*x[0] + prev1)**2)
            for t in range(2, H):
                cost += np.sum((x[t] - 2*x[t-1] + x[t-2])**2)
                
            # S2 & S3
            for t in range(H):
                if is_initial:
                    # S3 with weight 1
                    cost += np.sum((x[t] - A_target[t])**2)
                else:
                    w1 = np.exp(-self.alpha * t)
                    w2 = 1 - w1
                    # S2: Deviation from old
                    cost += w1 * np.sum((x[t] - A_old_future[t])**2)
                    # S3: Deviation from new
                    cost += w2 * np.sum((x[t] - A_target[t])**2)
            return cost

        # Velocity constraints
        # |x_t - x_{t-1}| <= v_max
        # We implement as inequality constraints: v_max - |delta| >= 0
        # => v_max^2 - delta^2 >= 0 (to avoid abs)
        
        cons = []
        def vel_con_0(x):
            x = x.reshape(H, -1)
            delta = x[0] - prev1
            return self.max_velocity**2 - delta**2
        cons.append({'type': 'ineq', 'fun': lambda x: vel_con_0(x).flatten()})
        
        for t in range(1, H):
            def vel_con(x, t=t):
                x = x.reshape(H, -1)
                delta = x[t] - x[t-1]
                return self.max_velocity**2 - delta**2
            cons.append({'type': 'ineq', 'fun': vel_con})

        # Optimization
        # SLSQP is good for constrained
        res = minimize(objective, x0, method='SLSQP', constraints=cons, tol=1e-3, options={'maxiter': 20})
        
        return res.x.reshape(H, -1)

    def get_action(self, obs):
        # Trigger inference if buffer is low
        # Assume chunk size is at least 20, trigger when < 10?
        # Or better: trigger when we have executed half of previous chunk.
        # But we don't know chunk size.
        # Let's use a fixed threshold.
        threshold = 10
        
        with self.lock:
            buffer_len = len(self.action_buffer)
            
        if buffer_len < threshold and not self.is_inferring:
            self.infer_async(obs)
            
        # Return action
        with self.lock:
            if self.action_buffer:
                action = self.action_buffer.popleft()
                self.executed_history.append(action)
                if self.is_inferring:
                    self.steps_since_inference_start += 1
                return action
            else:
                # Buffer empty. If inferring, wait?
                # For real-time, waiting is bad, but returning zeros is also bad.
                # Let's wait a bit.
                pass
        
        # If we are here, buffer is empty.
        while self.is_inferring:
            time.sleep(0.001)
            with self.lock:
                if self.action_buffer:
                    action = self.action_buffer.popleft()
                    self.executed_history.append(action)
                    return action
                    
        return np.zeros(self.action_dim) if self.action_dim else None
