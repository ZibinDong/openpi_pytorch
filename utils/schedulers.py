import math


class CosineDecaySchedule:
    """
    Implements a learning rate schedule with a linear warmup followed by a cosine decay.

    Args:
        warmup_steps (int): The number of steps for the linear warmup phase.
        peak_lr (float): The learning rate at the end of the warmup phase and the start of decay.
        decay_steps (int): The total number of steps over which the cosine decay occurs.
                           This typically includes the warmup steps if decay starts after warmup.
        decay_lr (float): The minimum learning rate at the end of the cosine decay.
                          If not specified, it defaults to 0.0.
    """

    def __init__(
        self, warmup_steps: int, peak_lr: float, decay_steps: int, decay_lr: float = 0.0
    ):
        if warmup_steps >= decay_steps:
            raise ValueError("warmup_steps must be less than decay_steps")
        if not (0 <= decay_lr <= peak_lr):
            raise ValueError("decay_lr must be between 0 and peak_lr")

        self.warmup_steps = warmup_steps
        self.peak_lr = peak_lr
        self.decay_steps = decay_steps
        self.decay_lr = decay_lr
        self.total_decay_range = (
            peak_lr - decay_lr
        )  # The range over which LR actually decays

    def __call__(self, current_step: int) -> float:
        if current_step < self.warmup_steps:
            # Linear Warmup Phase
            # Learning rate increases linearly from 0 to peak_lr
            return self.peak_lr * (current_step / self.warmup_steps)
        elif current_step < self.decay_steps:
            # Cosine Decay Phase
            # Calculate the progress within the decay period (after warmup)
            # The cosine decay starts from peak_lr and goes down to decay_lr
            progress = (current_step - self.warmup_steps) / (
                self.decay_steps - self.warmup_steps
            )

            # Cosine annealing formula: 0.5 * (1 + cos(pi * progress))
            # This scales from 1.0 down to 0.0
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))

            # Scale the cosine factor by the total_decay_range and add decay_lr
            # to ensure it decays from peak_lr to decay_lr
            return self.decay_lr + self.total_decay_range * cosine_factor
        else:
            # After decay_steps, learning rate stays at decay_lr
            return self.decay_lr
