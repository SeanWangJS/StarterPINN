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
    # Generate extra crack/sym points
    pts_crack_face = sampler.sample_crack_face(tc['crack_face_points']).numpy()
    pts_sym_axis = sampler.sample_symmetry_axis(tc['sym_axis_points']).numpy()
    
    # Import evaluator for elliptic mapping
    from src.evaluator import cartesian_to_elliptic
    
    def to_elliptic(pts_phys):
        pts_nd = torch.tensor(pts_phys / a, dtype=torch.float32, device=device)
        mu, nu = cartesian_to_elliptic(pts_nd, a_nd=1.0)
        return mu.numpy(), nu.numpy()

    mu_domain, nu_domain = to_elliptic(pts_domain)
    mu_crack, nu_crack = to_elliptic(pts_crack)
    mu_cf, nu_cf = to_elliptic(pts_crack_face)
    mu_sym, nu_sym = to_elliptic(pts_sym_axis)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # ================= Cartesian Plot =================
    ax1.scatter(pts_domain[:, 0], pts_domain[:, 1], s=1, c='blue', alpha=0.3, label='Domain Points')
    
    bc_colors = {'top': 'red', 'bottom': 'green', 'left': 'orange', 'right': 'purple'}
    for key, pts in bc_dict.items():
        if key == 'center': continue
        pts_np = pts.numpy()
        ax1.scatter(pts_np[:, 0], pts_np[:, 1], s=10, c=bc_colors[key], label=f'Boundary ({key})')
        
    pts_center = bc_dict['center'].numpy()
    ax1.scatter(pts_center[:, 0], pts_center[:, 1], s=50, c='black', marker='x', label='Center Fix')
    ax1.scatter(pts_crack[:, 0], pts_crack[:, 1], s=15, c='cyan', marker='*', label='Initial Crack ($H_0$)')
    ax1.scatter(pts_crack_face[:, 0], pts_crack_face[:, 1], s=10, c='magenta', marker='x', label='Crack Face')
    ax1.scatter(pts_sym_axis[:, 0], pts_sym_axis[:, 1], s=10, c='brown', marker='+', label='Sym Axis')
    
    ax1.set_xlim([-W/2 * 1.1, W/2 * 1.1])
    ax1.set_ylim([-H/2 * 1.1, H/2 * 1.1])
    ax1.set_aspect('equal')
    ax1.set_xlabel('x [mm]')
    ax1.set_ylabel('y [mm]')
    ax1.set_title('Collocation Points (Cartesian $x-y$)')
    ax1.legend(loc='upper right')
    
    # ================= Elliptic Plot =================
    ax2.scatter(mu_domain, nu_domain, s=1, c='blue', alpha=0.3, label='Domain Points')
    
    for key, pts in bc_dict.items():
        if key == 'center': continue
        pts_np = pts.numpy()
        mu_bc, nu_bc = to_elliptic(pts_np)
        ax2.scatter(mu_bc, nu_bc, s=10, c=bc_colors[key], label=f'Boundary ({key})')
        
    mu_center, nu_center = to_elliptic(pts_center)
    ax2.scatter(mu_center, nu_center, s=50, c='black', marker='x', label='Center Fix')
    ax2.scatter(mu_crack, nu_crack, s=15, c='cyan', marker='*', label='Initial Crack ($H_0$)')
    ax2.scatter(mu_cf, nu_cf, s=10, c='magenta', marker='x', label='Crack Face')
    ax2.scatter(mu_sym, nu_sym, s=10, c='brown', marker='+', label='Sym Axis')
    
    ax2.set_xlabel('$\mu$')
    ax2.set_ylabel('$\\nu$')
    ax2.set_title('Collocation Points (Elliptic $\mu-\\nu$)')
    ax2.legend(loc='upper right')
    
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'collocation_points.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved point distribution plot to {save_path}")

if __name__ == '__main__':
    plot_collocation_points()
