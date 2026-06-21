import torch
import torch.nn as nn

class LossWeightingStrategy(nn.Module):
    """
    Base class for loss weighting strategies.
    """
    def __init__(self):
        super().__init__()

    def compute_weighted_loss(self, raw_losses: dict, step: int = 0) -> tuple[torch.Tensor, dict]:
        """
        Calculates the weighted loss.
        
        Args:
            raw_losses: Dictionary containing raw loss components (e.g., 'pde', 'hole', 'sym', 'far').
            step: Current training step (useful for dynamic weighting schemes).
            
        Returns:
            total_loss: The scalar total weighted loss tensor.
            weights: Dictionary containing the weights applied to each component.
        """
        raise NotImplementedError


class ConstantWeighting(LossWeightingStrategy):
    """
    Static/Constant loss weighting strategy.
    """
    def __init__(self, w_pde=1.0, w_hole=1.0, w_sym=1.0, w_far=1.0):
        super().__init__()
        self.w_pde = w_pde
        self.w_hole = w_hole
        self.w_sym = w_sym
        self.w_far = w_far

    def compute_weighted_loss(self, raw_losses: dict, step: int = 0) -> tuple[torch.Tensor, dict]:
        total_loss = (
            self.w_pde * raw_losses['pde'] +
            self.w_hole * raw_losses['hole'] +
            self.w_sym * raw_losses['sym'] +
            self.w_far * raw_losses['far']
        )
        weights = {
            'w_pde': self.w_pde,
            'w_hole': self.w_hole,
            'w_sym': self.w_sym,
            'w_far': self.w_far
        }
        return total_loss, weights

class LRAWeighting(LossWeightingStrategy):
    """
    Learning Rate Annealing (LRA) dynamic weighting strategy.
    Reference: Wang et al., 2021.
    """
    def __init__(self, model: nn.Module, alpha: float = 0.9, update_freq: int = 10, target_keys=None):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.update_freq = update_freq
        self.target_keys = target_keys if target_keys else ['pde', 'hole', 'sym', 'far']
        
        self.weights_dict = nn.ParameterDict({
            k: nn.Parameter(torch.tensor(1.0), requires_grad=False) for k in self.target_keys
        })

    def compute_weighted_loss(self, raw_losses: dict, step: int = 0) -> tuple[torch.Tensor, dict]:
        if step > 0 and step % self.update_freq == 0:
            target_layer = self.model.net[-1].weight
            
            grads = {}
            with torch.enable_grad():
                for k in self.target_keys:
                    loss_k = raw_losses[k]
                    grad_k = torch.autograd.grad(loss_k, target_layer, retain_graph=True, allow_unused=True)[0]
                    grads[k] = grad_k if grad_k is not None else torch.zeros_like(target_layer)
                        
            if 'pde' in grads:
                max_grad_pde = torch.max(torch.abs(grads['pde'])).detach()
                for k in self.target_keys:
                    if k == 'pde': continue
                    mean_grad_k = torch.mean(torch.abs(grads[k])).detach()
                    if mean_grad_k > 1e-8:
                        hat_w_k = max_grad_pde / (mean_grad_k + 1e-8) # 防止除零
                        hat_w_k = torch.clamp(hat_w_k, min=0.1, max=100.0)                        
                        # hat_w_k = max_grad_pde / mean_grad_k
                        w_k_new = (1 - self.alpha) * self.weights_dict[k] + self.alpha * hat_w_k
                        self.weights_dict[k].copy_(w_k_new)

        total_loss = 0.0
        out_weights = {}
        for k in self.target_keys:
            w = self.weights_dict[k]
            total_loss = total_loss + w * raw_losses[k]
            out_weights[f'w_{k}'] = w.item()
            
        return total_loss, out_weights


class SoftAdaptWeighting(LossWeightingStrategy):
    """
    SoftAdapt dynamic weighting strategy.
    Reference: Heydari et al., 2019.
    """
    def __init__(self, beta: float = 0.1, target_keys=None):
        super().__init__()
        self.beta = beta
        self.target_keys = target_keys if target_keys else ['pde', 'hole', 'sym', 'far']
        
        self.weights_dict = nn.ParameterDict({
            k: nn.Parameter(torch.tensor(1.0), requires_grad=False) for k in self.target_keys
        })
        self.prev_losses = nn.ParameterDict({
            k: nn.Parameter(torch.tensor(-1.0), requires_grad=False) for k in self.target_keys
        })

    def compute_weighted_loss(self, raw_losses: dict, step: int = 0) -> tuple[torch.Tensor, dict]:
        if step > 0:
            rates = {}
            for k in self.target_keys:
                prev = self.prev_losses[k]
                if prev.item() > 0:
                    rates[k] = raw_losses[k].detach() / prev
                else:
                    rates[k] = torch.tensor(1.0, device=raw_losses[k].device)
            
            keys = sorted(self.target_keys)
            rate_tensor = torch.stack([rates[k] for k in keys])
            soft_weights = torch.softmax(self.beta * rate_tensor, dim=0) * len(keys)
            
            for i, k in enumerate(keys):
                self.weights_dict[k].copy_(soft_weights[i])

        for k in self.target_keys:
            self.prev_losses[k].copy_(raw_losses[k].detach())

        total_loss = 0.0
        out_weights = {}
        for k in self.target_keys:
            w = self.weights_dict[k]
            total_loss = total_loss + w * raw_losses[k]
            out_weights[f'w_{k}'] = w.item()
            
        return total_loss, out_weights
