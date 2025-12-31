import grasp_cube.real
import gymnasium as gym
import tyro
import dataclasses
import matplotlib.pyplot as plt
import numpy as np
from grasp_cube.real.fake_lerobot_env import FakeLeRobotEnvConfig
from env_client import websocket_client_policy as _websocket_client_policy
from grasp_cube.real import MonitorWrapper, EvalRecordConfig, EvalRecordWrapper
import cvxpy as cp
import time
import threading

@dataclasses.dataclass
class RTGConfig:
    """Configuration for Real-time Trajectory Generation Module"""
    lambda_smooth: float = 5.0  # Weight for smoothing term (S1)
    lambda_old: float = 10.0  # Weight for old trajectory deviation (S2)
    lambda_new: float = 5.0  # Weight for new trajectory deviation (S3)
    decay_rate: float = 5.0  # Exponential decay rate for W1(t)
    velocity_limit: float = 20.0  # Joint velocity limit (rad/s or m/s)
    dt: float = 0.1  # Time step between actions (control frequency)
    
@dataclasses.dataclass
class Args:
    env: FakeLeRobotEnvConfig 
    eval: EvalRecordConfig
    rtg: RTGConfig = dataclasses.field(default_factory=RTGConfig)
    host: str = "0.0.0.0"
    port: int = 8000
    monitor_host: str = "0.0.0.0"
    monitor_port: int = 9000
    num_episodes: int = 10
    control_frequency: float = 10.0  # Hz - fixed control frequency for stepping environment

class AsyncPolicyClient:
    """
    Asynchronous policy client that performs inference in a background thread.
    
    Key design:
    - Main thread runs the environment at a fixed control frequency
    - Background thread handles blocking inference calls
    - Uses locks to protect shared state
    """
    
    def __init__(self, client: _websocket_client_policy.WebsocketClientPolicy):
        self.client = client
        self.inference_thread = None
        self.lock = threading.Lock()
        
        # Shared state (needs lock protection)
        self.is_inferring = False
        self.latest_chunk = None
        self.inference_time = None
        self.obs_to_infer = None
        
    def start_inference(self, obs: dict):
        """
        Start asynchronous inference (non-blocking).
        
        If inference is already in progress, ignore this request (to avoid queuing).
        """
        with self.lock:
            if self.is_inferring:
                return  # Already inferring, skip
            self.is_inferring = True
            self.obs_to_infer = obs
        
        # Start background thread
        self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.inference_thread.start()
    
    def _inference_worker(self):
        """Background worker thread: performs blocking inference"""
        # Get observation
        with self.lock:
            obs = self.obs_to_infer
        
        # Perform blocking inference (without holding lock)
        start_time = time.monotonic()
        try:
            response = self.client.infer(obs)
            inference_time = time.monotonic() - start_time
            chunk = np.array(response["action"])
        except Exception as e:
            print(f"Inference error: {e}")
            with self.lock:
                self.is_inferring = False
            return
        
        # Update results
        with self.lock:
            self.latest_chunk = chunk
            self.inference_time = inference_time
            self.inference_start_time = start_time
            self.is_inferring = False
    
    def try_get_chunk(self):
        """
        Try to get the latest inference result (non-blocking).
        
        Returns:
            (chunk, inference_time): If there is a new result, otherwise (None, None)
        """
        with self.lock:
            if self.latest_chunk is not None:
                chunk = self.latest_chunk
                inference_time = self.inference_time
                # Clear to avoid reuse
                self.latest_chunk = None
                self.inference_time = None
                return chunk, inference_time
        return None, None
    
    def is_busy(self) -> bool:
        """Check if currently inferring"""
        with self.lock:
            return self.is_inferring
    
    def reset(self):
        """Reset the client"""
        self.client.reset()
        with self.lock:
            self.is_inferring = False
            self.latest_chunk = None
            self.inference_time = None

class RTGTrajectoryGenerator:
    """
    Real-time Trajectory Generation Module using QP optimization.
    
    This module implements the RTG method from arXiv:2507.17141v1, which:
    1. Discards outdated portions of new action chunks based on inference latency
    2. Smoothly blends new chunks with currently executing trajectory using QP
    3. Ensures continuity, smoothness, and respects velocity constraints
    """
    
    def __init__(self, config: RTGConfig):
        self.config = config
        self.current_trajectory = None
        self.current_index = 0
        self.trajectory_start_time = None
        self.action_dim = None  # To be set on first update

    def set_action_dim(self, action_dim: int):
        """Set the action dimension"""
        self.action_dim = action_dim
        
    def reset(self):
        """Reset the trajectory generator state"""
        self.current_trajectory = None
        self.current_index = 0
        self.trajectory_start_time = None
    
    def update_trajectory(self, new_chunk: np.ndarray, inference_latency: float):
        """
        Update trajectory with new action chunk using RTG blending.
        
        Args:
            new_chunk: New action chunk from policy [chunk_size, action_dim]
            inference_latency: Time taken for inference (t1)
        
        Key logic:
        1. Calculate the time elapsed from observation to now (t1 + t2)
        2. Discard A_new[0 : t1+t2]
        3. Use QP to blend A_new[t1+t2:] and A_old[t1+t2:]
        """
        processing_start = time.monotonic()
        
        if self.current_trajectory is None:
            # First trajectory - no blending needed, but discard outdated portion
            elapsed = inference_latency
            delay_steps = int(elapsed / self.config.dt)
            if delay_steps < len(new_chunk):
                self.current_trajectory = new_chunk[delay_steps:]
            else:
                # Entire chunk is outdated, take only the last action
                self.current_trajectory = new_chunk[-1:].repeat(len(new_chunk) - delay_steps, axis=0)
            self.current_index = 0
            self.trajectory_start_time = time.monotonic()
        else:
            # Calculate processing time t2
            processing_time = time.monotonic() - processing_start
            total_delay = inference_latency + processing_time
            
            # Perform RTG blending
            self.current_trajectory = self._blend_trajectories(
                new_chunk=new_chunk,
                total_delay=total_delay
            )
            self.current_index = 0
            self.trajectory_start_time = time.monotonic()
    
    def needs_new_chunk(self, lookahead_steps: int = 5) -> bool:
        """
        Determine if a new action chunk is needed.
        
        Args:
            lookahead_steps: How many steps ahead to trigger new inference
        
        Returns:
            True if we need to request a new chunk
        """
        if self.current_trajectory is None:
            return True
        remaining = len(self.current_trajectory) - self.current_index
        return remaining <= lookahead_steps
    
    def _blend_trajectories(self, new_chunk: np.ndarray, total_delay: float) -> np.ndarray:
        """
        Blend new action chunk with current trajectory using QP optimization.
        
        Following RTG algorithm:
        1. Discard A_new[0 : t1+t2] (outdated portion)
        2. Continue with A_old[0 : t1+t2]
        3. Blend A_new[t1+t2 : te] with A_old[t1+t2 : tf] using QP
        """
        dt = self.config.dt
        delay_steps = int(total_delay / dt)
        
        # Discard outdated portion of new chunk
        if delay_steps >= len(new_chunk):
            # Entire new chunk is outdated, keep executing old trajectory
            return self.current_trajectory[self.current_index:]
        
        new_chunk_valid = new_chunk[delay_steps:]
        
        # Get remaining portion of old trajectory
        old_remaining = self.current_trajectory[self.current_index:]
        
        if len(old_remaining) == 0:
            # Old trajectory exhausted, use new chunk
            return new_chunk_valid
        
        # Determine blending horizon
        blend_length = min(len(new_chunk_valid), len(old_remaining))
        
        if blend_length <= 1:
            # Not enough data to blend, concatenate
            if len(old_remaining) > 0:
                return np.concatenate([old_remaining[:1], new_chunk_valid])
            return new_chunk_valid
        
        # Prepare trajectories for blending
        old_blend = old_remaining[:blend_length]
        new_blend = new_chunk_valid[:blend_length]
        action_dim = new_chunk.shape[1]
        
        # Solve QP for blended trajectory
        blended = self._solve_qp_blend(old_blend, new_blend, blend_length, action_dim)
        
        # Concatenate: blended portion + rest of new chunk
        if len(new_chunk_valid) > blend_length:
            result = np.concatenate([blended, new_chunk_valid[blend_length:]])
        else:
            result = blended
            
        return result
    
    def _solve_qp_blend(
        self, 
        old_traj: np.ndarray, 
        new_traj: np.ndarray, 
        T: int, 
        action_dim: int
    ) -> np.ndarray:
        """
        Solve QP optimization for trajectory blending.
        
        Objective:
            min: λ_smooth * S1 + λ_old * S2 + λ_new * S3
        where:
            S1: smoothing term (minimize accelerations)
            S2: deviation from old trajectory with time-decaying weight W1(t)
            S3: deviation from new trajectory with time-increasing weight W2(t) = 1 - W1(t)
        
        Constraints:
            C1: velocity limits
        """
        # Decision variables: trajectory [T, action_dim]
        A = cp.Variable((T, action_dim))
        
        # Time-dependent weights
        t_normalized = np.linspace(0, 1, T)
        W1 = np.exp(-self.config.decay_rate * t_normalized)  # Decaying weight for old
        W2 = 1 - W1  # Increasing weight for new
        
        # S1: Smoothing term (minimize accelerations)
        # Acceleration approximated as: a[t] = (x[t+1] - 2*x[t] + x[t-1]) / dt^2
        smoothing_cost = 0
        if T > 2:
            for t in range(1, T - 1):
                accel = (A[t+1] - 2 * A[t] + A[t-1]) / (self.config.dt ** 2)
                smoothing_cost += cp.sum_squares(accel)
        
        # S2: Deviation from old trajectory (time-decaying)
        old_deviation_cost = 0
        for t in range(T):
            deviation = A[t] - old_traj[t]
            old_deviation_cost += W1[t] * cp.sum_squares(deviation)
        
        # S3: Deviation from new trajectory (time-increasing)
        new_deviation_cost = 0
        for t in range(T):
            deviation = A[t] - new_traj[t]
            new_deviation_cost += W2[t] * cp.sum_squares(deviation)
        
        # Total objective
        objective = cp.Minimize(
            self.config.lambda_smooth * smoothing_cost +
            self.config.lambda_old * old_deviation_cost +
            self.config.lambda_new * new_deviation_cost
        )
        
        # C1: Velocity constraints
        constraints = []
        for t in range(T - 1):
            velocity = (A[t+1] - A[t]) / self.config.dt
            constraints.append(velocity <= self.config.velocity_limit)
            constraints.append(velocity >= -self.config.velocity_limit)
        
        # Boundary condition: start from current position (continuity)
        constraints.append(A[0] == old_traj[0])
        
        # Solve QP
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-4, eps_rel=1e-4)
            if A.value is None or problem.status not in ["optimal", "optimal_inaccurate"]:
                print(f"QP optimization failed with status: {problem.status}, using linear blend")
                return self._linear_blend(old_traj, new_traj, T)
            return A.value
        except Exception as e:
            print(f"QP solver error: {e}, using linear blend")
            return self._linear_blend(old_traj, new_traj, T)
    
    def _linear_blend(self, old_traj: np.ndarray, new_traj: np.ndarray, T: int) -> np.ndarray:
        """Fallback: simple linear interpolation between old and new trajectories"""
        alpha = np.linspace(0, 1, T).reshape(-1, 1)
        return (1 - alpha) * old_traj + alpha * new_traj
    
    def get_next_action(self) -> np.ndarray:
        """Get the next action from current trajectory"""
        if self.current_trajectory is None or self.current_index >= len(self.current_trajectory):
            raise RuntimeError("No trajectory available")
        
        action = self.current_trajectory[self.current_index]
        self.current_index += 1
        return action
    
    def get_last_action(self) -> np.ndarray:
        """
        Get the last action (for emergency when trajectory is exhausted).
        
        Returns:
            Last action in the trajectory, or zero action if no trajectory
        """
        if self.current_trajectory is None or len(self.current_trajectory) == 0:
            return np.zeros(self.action_dim)
        return self.current_trajectory[-1]
    
    def has_actions(self) -> bool:
        """Check if there are remaining actions in current trajectory"""
        return (self.current_trajectory is not None and 
                self.current_index < len(self.current_trajectory))

def main(args: Args):
    env = gym.make(
        "FakeLeRobotEnv-v0",
        config=args.env,
    )
    env = MonitorWrapper(env, port=args.monitor_port, host=args.monitor_host, include_images=True)
    env = EvalRecordWrapper(env, config=args.eval)
    
    # Initialize client and RTG
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    async_client = AsyncPolicyClient(client)
    rtg = RTGTrajectoryGenerator(args.rtg)
    rtg.set_action_dim(env.action_space.shape[0])
    
    # Control frequency parameters
    dt = 1.0 / args.control_frequency  # Control period
    
    print(f"Starting RTG evaluation for {args.num_episodes} episodes")
    print(f"Control frequency: {args.control_frequency} Hz (dt={dt:.3f}s)")
    print(f"RTG Config: λ_smooth={args.rtg.lambda_smooth}, λ_old={args.rtg.lambda_old}, "
          f"λ_new={args.rtg.lambda_new}, decay_rate={args.rtg.decay_rate}")
    
    for episode in range(args.num_episodes):
        obs, info = env.reset()
        async_client.reset()
        rtg.reset()
        
        done = False
        actions = []
        gt_actions = []
        inference_times = []
        step_count = 0
        loop_times = []
        
        # Immediately start the first inference
        async_client.start_inference(obs)
        first_inference_out = False
        
        # Main control loop - run at fixed frequency
        while not done:
            loop_start = time.monotonic()
            
            # ========================================
            # 1. Check if there are new inference results
            # ========================================
            new_chunk, inference_time = async_client.try_get_chunk()
            if new_chunk is not None:
                # Received new action chunk, update RTG trajectory
                first_inference_out = True
                inference_times.append(inference_time)
                rtg.update_trajectory(new_chunk, inference_time)
                print(f"  [Step {step_count}] New chunk received, inference time: {inference_time*1000:.2f}ms")
            if not first_inference_out:
                # Wait for the first inference result
                time.sleep(0.01)
                continue
            # ========================================
            # 2. Determine if new inference is needed
            # ========================================
            if not async_client.is_busy() and rtg.needs_new_chunk(lookahead_steps=10):
                # Need new action chunk and no inference in progress, start async inference
                async_client.start_inference(obs)
            
            # ========================================
            # 3. Get the action to execute from RTG
            # ========================================
            if rtg.has_actions():
                action = rtg.get_next_action()
            else:
                # Emergency when trajectory is exhausted: use last action
                action = rtg.get_last_action()
                print(f"  [Step {step_count}] WARNING: Trajectory exhausted, using last action")
            
            # ========================================
            # 4. Execute action
            # ========================================
            obs, reward, done, truncated, info = env.step(action)
            actions.append(action)
            step_count += 1
            
            gt_action = info.get("gt_action")
            assert gt_action is not None, "Ground truth action missing in info"
            gt_actions.append(gt_action)
            
            # ========================================
            # 5. Control fixed frequency
            # ========================================
            loop_elapsed = time.monotonic() - loop_start
            loop_times.append(loop_elapsed)
            sleep_time = dt - loop_elapsed
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif loop_elapsed > dt * 1.5:
                # If loop time exceeds expected by too much, issue warning
                print(f"  [Step {step_count}] WARNING: Loop time {loop_elapsed*1000:.2f}ms "
                      f"exceeds target {dt*1000:.2f}ms")
        
        # ========================================
        # Visualization and statistics
        # ========================================
        actions = np.array(actions)
        gt_actions = np.array(gt_actions)
        steps = np.arange(len(actions))
        num_actions = actions.shape[1]
        
        # Plot actions comparison
        fig, axs = plt.subplots(num_actions, 1, figsize=(12, 3 * num_actions))
        if num_actions == 1:
            axs = [axs]
        
        for i in range(num_actions):
            axs[i].plot(steps, actions[:, i], label="RTG Predicted Action", 
                       linewidth=2, alpha=0.8)
            if gt_actions[:, i].any():
                axs[i].plot(steps, gt_actions[:, i], label="Ground Truth Action", 
                           linestyle='--', linewidth=1.5, alpha=0.7)
            axs[i].set_xlabel("Step")
            axs[i].set_ylabel(f"Action Dim {i}")
            axs[i].legend()
            axs[i].grid(alpha=0.3)
        
        plt.suptitle(f"Episode {episode} - RTG Trajectory Generation")
        plt.tight_layout()
        plt.savefig(env.run_dir / f"episode_{episode}_actions.png", dpi=150)
        plt.close(fig)
        
        # Print episode statistics
        avg_inference_time = np.mean(inference_times) if inference_times else 0
        max_inference_time = np.max(inference_times) if inference_times else 0
        num_inferences = len(inference_times)
        avg_loop_time = np.mean(loop_times)
        max_loop_time = np.max(loop_times)
        
        print(f"\nEpisode {episode} Summary:")
        print(f"  Steps: {step_count}, Inferences: {num_inferences}")
        print(f"  Inference time - avg: {avg_inference_time*1000:.2f}ms, max: {max_inference_time*1000:.2f}ms")
        print(f"  Loop time - avg: {avg_loop_time*1000:.2f}ms, max: {max_loop_time*1000:.2f}ms, "
              f"target: {dt*1000:.2f}ms")
        
        # Check if inference time meets RTG requirements
        if num_inferences > 0:
            chunk_duration = len(new_chunk) * args.rtg.dt if new_chunk is not None else 0
            if max_inference_time > chunk_duration / 2:
                print(f"  ⚠ WARNING: Max inference time ({max_inference_time*1000:.2f}ms) exceeds "
                      f"half chunk duration ({chunk_duration*500:.2f}ms)")
            else:
                print(f"  ✓ Inference time within acceptable range (< {chunk_duration*500:.2f}ms)")
    
    print(f"\nEvaluation complete. Results saved to {env.run_dir}")

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)

