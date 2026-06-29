import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import sys

# Add parent directory to path to allow src imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.geometry import GriffithGeometrySampler

def plot_collocation_points():
    device = torch.device('cpu')
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    phys = cfg['physics']
    W, H, a = phys['W'], phys['H'], phys['a']
    
    tc = cfg['train']
    
    # Initialize sampler
    sampler = GriffithGeometrySampler(W=W, H=H, a=a, device=device)
    
    # Generate points
    pts_domain = sampler.sample_fixed_domain(tc['domain_points']).numpy()
    bc_dict = sampler.sample_boundaries(tc['bc_points'])
    pts_crack = sampler.get_initial_crack_points(tc['crack_points']).numpy()
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Domain points
    ax.scatter(pts_domain[:, 0], pts_domain[:, 1], s=1, c='blue', alpha=0.3, label='Domain Points')
    
    # Boundary points
    bc_colors = {'top': 'red', 'bottom': 'green', 'left': 'orange', 'right': 'purple'}
    for key, pts in bc_dict.items():
        if key == 'center': continue
        pts_np = pts.numpy()
        ax.scatter(pts_np[:, 0], pts_np[:, 1], s=10, c=bc_colors[key], label=f'Boundary ({key})')
        
    # Center point
    pts_center = bc_dict['center'].numpy()
    ax.scatter(pts_center[:, 0], pts_center[:, 1], s=50, c='black', marker='x', label='Center Fix')
    
    # Crack points
    ax.scatter(pts_crack[:, 0], pts_crack[:, 1], s=15, c='cyan', marker='*', label='Initial Crack ($H_0$)')
    
    ax.set_xlim([-W/2 * 1.1, W/2 * 1.1])
    ax.set_ylim([-H/2 * 1.1, H/2 * 1.1])
    ax.set_aspect('equal')
    ax.set_title('Collocation Points Distribution')
    ax.legend(loc='upper right')
    
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'collocation_points.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved point distribution plot to {save_path}")

if __name__ == '__main__':
    plot_collocation_points()
