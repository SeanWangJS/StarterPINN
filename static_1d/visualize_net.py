import matplotlib.pyplot as plt

def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    """
    使用 matplotlib 绘制神经网络结构示意图
    """
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    
    # 绘制节点 (Nodes)
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/5.,
                                color='skyblue', ec='k', zorder=4)
            ax.add_artist(circle)
            
    # 绘制连线 (Edges)
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='gray', alpha=0.3)
                ax.add_artist(line)

if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    
    # 我们的真实网络结构是: 1 -> 20 -> 20 -> 20 -> 1
    # 为了绘图直观且不显得一团黑线，我们用 1 -> 8 -> 8 -> 8 -> 1 作为示意图，并加上文字说明
    draw_neural_net(ax, .1, .9, .1, .9, [1, 8, 8, 8, 1])
    
    # 添加标注
    plt.text(0.1, 0.95, 'Input Layer\n(1 node: x)', ha='center', fontsize=12, fontweight='bold')
    plt.text(0.36, 0.95, 'Hidden 1\n(20 nodes, Tanh)', ha='center', fontsize=10)
    plt.text(0.63, 0.95, 'Hidden 2\n(20 nodes, Tanh)', ha='center', fontsize=10)
    plt.text(0.9, 0.95, 'Output Layer\n(1 node: u)', ha='center', fontsize=12, fontweight='bold')
    
    plt.title('PINN MLP Architecture Schematic', fontsize=14, pad=20)
    
    import os
    save_path = os.path.join(os.path.dirname(__file__), 'mlp_architecture.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Network architecture saved to {save_path}")
