import numpy as np

class HybridSRLReplayBuffer:
    def __init__(self, capacity, mastery_threshold=0.9, initial_interval=1.0, multiplier=1.5, alpha=1.0, beta=0.5):
        """
        Hybrid SRL Replay Buffer with ExIt diversity weighting.
        
        Parameters:
        - capacity: Maximum number of experiences in the buffer.
        - mastery_threshold: Recall score threshold for considering an item 'mastered'.
        - initial_interval: Starting review interval for new items.
        - multiplier: Base multiplier for interval growth on successful recalls.
        - alpha: Strength of diversity weighting (higher = more aggressive shortening for diverse items).
        - beta: Blend factor between variance (beta) and embedding distance (1-beta) in diversity score.
        """
        self.buffer = []  # List of dicts: {'experience': any, 'last_review': int, 'interval': float, 'easiness': float, 'variance': float, 'embedding': np.array}
        self.capacity = capacity
        self.mastery_threshold = mastery_threshold
        self.initial_interval = initial_interval
        self.multiplier = multiplier
        self.alpha = alpha
        self.beta = beta

    def add(self, experience, embedding, variance=0.0):
        """
        Add a new experience to the buffer.
        
        Parameters:
        - experience: The experience data (e.g., string or tuple for POMDP trajectory).
        - embedding: np.array representing the embedding of the experience.
        - variance: Float for reward variance (from ExIt/GRPO).
        """
        if len(self.buffer) >= self.capacity:
            # Simple FIFO eviction; could improve with priority-based
            self.buffer.pop(0)
        self.buffer.append({
            'experience': experience,
            'last_review': 0,
            'interval': self.initial_interval,
            'easiness': 2.0,
            'variance': variance,
            'embedding': embedding
        })

    def compute_diversity(self, item):
        """
        Compute normalized diversity score for an item (blend of variance and embedding distance to centroid).
        """
        if not self.buffer:
            return 0.0
        embeddings = np.array([b['embedding'] for b in self.buffer])
        centroid = np.mean(embeddings, axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        max_dist = np.max(distances) if np.max(distances) > 0 else 1.0
        min_dist = np.min(distances)
        d_i = np.linalg.norm(item['embedding'] - centroid) / (max_dist - min_dist + 1e-6)
        normalized_diversity = self.beta * item['variance'] + (1 - self.beta) * d_i
        return normalized_diversity

    def get_due_samples(self, current_step, batch_size):
        """
        Get a batch of due experiences, prioritized by difficulty (harder first).
        
        Parameters:
        - current_step: Current global step or inference count.
        - batch_size: Number of samples to return.
        
        Returns:
        - List of due experience dicts.
        """
        due = [item for item in self.buffer if current_step >= item['last_review'] + item['interval']]
        # Prioritize harder items (lower easiness = higher priority)
        due.sort(key=lambda x: 1 / x['easiness'], reverse=True)  # Descending sort for harder first
        return due[:batch_size]

    def update_after_replay(self, items, recall_scores, current_step):
        """
        Update SRL metadata for replayed items based on recall scores and diversity.
        
        Parameters:
        - items: List of experience dicts that were replayed.
        - recall_scores: List of float scores (0-1) for each item's recall performance.
        - current_step: Current global step.
        """
        for item, score in zip(items, recall_scores):
            diversity = self.compute_diversity(item)
            if score >= self.mastery_threshold:
                item['easiness'] = min(item['easiness'] + 0.1, 2.5)
                base_interval = item['interval'] * self.multiplier * (item['easiness'] - 1)
            else:
                item['easiness'] = max(item['easiness'] - 0.2, 1.3)
                base_interval = 1.0  # Reset for hard items
            # Weight by diversity: shorter interval for higher diversity
            item['interval'] = base_interval / (1 + self.alpha * diversity)
            item['last_review'] = current_step

# Example Simulation
# Initialize buffer
buffer = HybridSRLReplayBuffer(capacity=10)

# Add dummy experiences with varying embeddings and variances
buffer.add("exp1: low div low var", np.array([0.0, 0.0]), variance=0.1)
buffer.add("exp2: med div med var", np.array([0.5, 0.5]), variance=0.3)
buffer.add("exp3: high div high var", np.array([10.0, 10.0]), variance=0.8)
buffer.add("exp4: low div low var", np.array([0.1, 0.1]), variance=0.05)
buffer.add("exp5: med-high div med var", np.array([5.0, 5.0]), variance=0.6)

# Print initial buffer
print("Initial Buffer:")
for item in buffer.buffer:
    div = buffer.compute_diversity(item)
    print(f"{item['experience']}: interval={item['interval']:.2f}, easiness={item['easiness']:.2f}, variance={item['variance']:.2f}, embedding={item['embedding']}, diversity={div:.2f}")

# Simulate updates at different steps
steps = [1, 5, 10]
recall_scores_list = [
    [0.7, 0.95, 0.85],  # Step 1: poor, good, marginal
    [0.88, 0.92, 0.96],  # Step 2: marginal, good, good
    [0.91, 0.75, 0.98]   # Step 3: good, poor, good
]

for step, recall_scores in zip(steps, recall_scores_list):
    print(f"\nSimulating Update at current_step={step}")
    due_items = buffer.get_due_samples(step, batch_size=3)
    if due_items:
        buffer.update_after_replay(due_items, recall_scores, step)
        print("Updated Buffer:")
        for item in buffer.buffer:
            div = buffer.compute_diversity(item)
            print(f"{item['experience']}: interval={item['interval']:.2f}, easiness={item['easiness']:.2f}, variance={item['variance']:.2f}, embedding={item['embedding']}, diversity={div:.2f}")
    else:
        print("No due items at this step.")
