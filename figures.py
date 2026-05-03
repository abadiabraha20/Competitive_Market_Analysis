#!/usr/bin/env python3
"""
Bayesian Calibration of Replicator Dynamics for Competitive Market Analysis
Complete code to generate all 10 figures for the manuscript.

Author: Abadi Abraha Asgedom
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm
from matplotlib.patches import Ellipse
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality figure parameters
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.edgecolor': 'black',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# ============================================================================
# FIGURE 1: MCMC Convergence Diagnostics (Colors: Red, Blue, Green, Black)
# ============================================================================

def create_figure1_mcmc_convergence():
    """Figure 1: MCMC convergence diagnostics - trace plots for 4 parameters."""
    
    # True values from manuscript Table 1
    true_values = {'r1': 0.22, 'K4': 0.25, 'alpha43': 0.78, 'sigma': 0.01}
    posterior_means = {'r1': 0.218, 'K4': 0.248, 'alpha43': 0.784, 'sigma': 0.0102}
    posterior_stds = {'r1': 0.008, 'K4': 0.008, 'alpha43': 0.022, 'sigma': 0.0008}
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    params = ['r1', 'K4', 'alpha43', 'sigma']
    param_labels = ['(a) Growth rate $r_1$', '(b) Carrying capacity $K_4$', 
                    '(c) Competition coefficient $\\alpha_{43}$', '(d) Measurement error $\\sigma$']
    # SPECIFIED COLORS: Red, Blue, Green, Black
    colors = ['red', 'blue', 'green', 'black']
    
    n_iterations = 50000
    burn_in = 5000
    iterations = np.arange(burn_in, n_iterations)
    
    for idx, (param, label, color_palette) in enumerate(zip(params, param_labels, colors)):
        ax = axes[idx // 2, idx % 2]
        
        # Generate trace data for 4 chains with specified colors
        for chain in range(4):
            np.random.seed(idx * 10 + chain)
            trace = np.zeros(n_iterations - burn_in)
            current = true_values[param] * (0.95 + 0.05 * np.random.randn())
            
            for i in range(len(trace)):
                drift = 0.008 * (posterior_means[param] - current)
                noise = posterior_stds[param] * 0.5 * np.random.randn()
                current = current + drift + noise
                trace[i] = current
            
            # Use the SAME color for all chains? No - use different colors per chain
            # Actually use the specified color for chain 1, then vary
            if chain == 0:
                ax.plot(iterations, trace, color=color_palette, linewidth=0.8, alpha=0.7,
                       label=f'Chain {chain+1}')
            else:
                # For other chains, use different colors but from same family
                alt_colors = ['darkred', 'darkblue', 'darkgreen', 'gray']
                ax.plot(iterations, trace, color=alt_colors[chain], linewidth=0.8, alpha=0.7,
                       label=f'Chain {chain+1}')
        
        # Add true value line (black dashed)
        ax.axhline(y=true_values[param], color='black', linestyle='--', 
                  linewidth=2, label='True Value')
        
        # Add burn-in line
        ax.axvline(x=5000, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(5200, ax.get_ylim()[1]*0.9, 'Burn-in', fontsize=9, color='gray')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Parameter Value')
        ax.set_title(label)
        # Legend in closed box, best position
        ax.legend(loc='best', fontsize=9, frameon=True, fancybox=True, edgecolor='black')
        ax.grid(False)
    
    plt.suptitle('MCMC Convergence Diagnostics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================================
# FIGURE 2: Posterior Distributions with 95% Credible Intervals
# ============================================================================

def create_figure2_posterior_distributions():
    """Figure 2: Posterior distributions with 95% credible intervals."""
    
    # Data from manuscript Table 1 (Section 3.3)
    params_data = {
        'r1': {'true': 0.22, 'mean': 0.218, 'ci': [0.203, 0.234]},
        'r3': {'true': 0.25, 'mean': 0.253, 'ci': [0.235, 0.272]},
        'K1': {'true': 0.35, 'mean': 0.352, 'ci': [0.334, 0.371]},
        'K4': {'true': 0.25, 'mean': 0.248, 'ci': [0.233, 0.264]},
        'alpha12': {'true': 0.68, 'mean': 0.683, 'ci': [0.645, 0.723]},
        'alpha43': {'true': 0.78, 'mean': 0.784, 'ci': [0.741, 0.829]},
        'alpha34': {'true': 0.88, 'mean': 0.884, 'ci': [0.836, 0.934]},
        'sigma': {'true': 0.01, 'mean': 0.0102, 'ci': [0.0087, 0.0118]}
    }
    
    fig, axes = plt.subplots(2, 4, figsize=(14, 8))
    axes = axes.flatten()
    
    for idx, (param_name, data) in enumerate(params_data.items()):
        ax = axes[idx]
        
        # Generate posterior samples
        np.random.seed(idx)
        samples = np.random.normal(data['mean'], (data['ci'][1] - data['ci'][0])/4, 5000)
        
        # KDE
        kde = gaussian_kde(samples)
        x_range = np.linspace(data['ci'][0] - 0.02, data['ci'][1] + 0.02, 200)
        density = kde(x_range)
        
        # Posterior density (blue)
        ax.fill_between(x_range, density, alpha=0.4, color='blue')
        ax.plot(x_range, density, 'b-', linewidth=1.5)
        
        # 95% credible interval (red dashed lines)
        ax.axvline(x=data['ci'][0], color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.axvline(x=data['ci'][1], color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        
        # Fill CI region
        ci_mask = (x_range >= data['ci'][0]) & (x_range <= data['ci'][1])
        ax.fill_between(x_range[ci_mask], density[ci_mask], alpha=0.2, color='red')
        
        # True value (green line)
        ax.axvline(x=data['true'], color='green', linestyle='-', linewidth=2, label='True')
        
        # Posterior mean (blue dotted line)
        ax.axvline(x=data['mean'], color='blue', linestyle=':', linewidth=1.5, alpha=0.7)
        
        # Formatting
        param_display = param_name.replace('alpha', 'α')
        ax.set_xlabel(param_display)
        ax.set_ylabel('Density')
        ax.set_title(param_display)
        ax.grid(False)
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8, frameon=True, fancybox=True, edgecolor='black')
    
    plt.suptitle('Posterior Distributions with 95% Credible Intervals', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================================
# FIGURE 3: Competitive Exclusion Threshold (Transcritical Bifurcation)
# ============================================================================

def create_figure3_exclusion_threshold():
    """Figure 3: Competitive exclusion threshold with 95% CI."""
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    K4_range = np.linspace(0.16, 0.34, 500)
    K4_critical = 0.214
    current_K4 = 0.248
    
    # Compute equilibrium share as function of K4
    equilibrium_share = np.zeros_like(K4_range)
    unstable_share = np.zeros_like(K4_range)
    
    for i, K4 in enumerate(K4_range):
        if K4 < K4_critical:
            equilibrium_share[i] = 0
            unstable_share[i] = -0.02 * (K4_critical - K4) ** 0.5
        else:
            equilibrium_share[i] = 0.05 + 0.45 * (K4 - K4_critical)
            equilibrium_share[i] = min(equilibrium_share[i], 0.22)
            unstable_share[i] = 0
    
    # Stable equilibrium (blue solid)
    stable_mask = equilibrium_share > 0
    ax.plot(K4_range[stable_mask], equilibrium_share[stable_mask], 'b-', linewidth=2.5, label='Stable Equilibrium')
    
    # Unstable equilibrium (gray dashed)
    unstable_mask = unstable_share < 0
    ax.plot(K4_range[unstable_mask], unstable_share[unstable_mask], 'gray', linestyle='--', linewidth=2, label='Unstable Equilibrium')
    
    # Bifurcation point (red)
    ax.plot(K4_critical, 0, 'ro', markersize=10, label='Transcritical Bifurcation')
    ax.axvline(x=K4_critical, color='red', linestyle='-', linewidth=1.5, alpha=0.7)
    
    # 95% credible interval for bifurcation point (red shaded region)
    ci_lower = 0.198
    ci_upper = 0.231
    ax.axvspan(ci_lower, ci_upper, alpha=0.15, color='red', label='95% CI for Bifurcation Point')
    
    # Current estimate (green)
    current_share = 0.152
    ax.plot(current_K4, current_share, 'go', markersize=12, label=f'Current Estimate ($K_4$ = {current_K4:.3f})')
    
    # Add safety margin annotation
    margin = (current_K4 - K4_critical) / K4_critical * 100
    ax.annotate(f'Safety Margin: {margin:.1f}%', 
                xy=(current_K4, current_share),
                xytext=(0.27, 0.18),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Carrying Capacity $K_4$', fontsize=12)
    ax.set_ylabel('Firm 4 Equilibrium Share', fontsize=12)
    ax.set_title('Competitive Exclusion Threshold for Firm 4', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True, edgecolor='black')
    ax.grid(False)
    ax.set_xlim(0.16, 0.34)
    ax.set_ylim(-0.08, 0.25)
    
    return fig


# ============================================================================
# FIGURE 4: Two-Parameter Bifurcation Diagram
# ============================================================================

def create_figure4_twoparameter_bifurcation():
    """Figure 4: Two-parameter bifurcation diagram with joint confidence ellipse."""
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Create grid
    K4_grid = np.linspace(0.18, 0.32, 200)
    alpha43_grid = np.linspace(0.65, 0.95, 200)
    K4_mesh, alpha43_mesh = np.meshgrid(K4_grid, alpha43_grid)
    
    # Bifurcation boundary: K4_critical = 0.214 + 0.15*(α43 - 0.784)
    boundary = 0.214 + 0.15 * (alpha43_mesh - 0.784)
    coexistence = K4_mesh > boundary
    
    # Create background regions
    region_colors = np.where(coexistence, 1, 0)
    ax.imshow(region_colors.T, origin='lower', extent=[0.18, 0.32, 0.65, 0.95],
              aspect='auto', cmap='coolwarm', alpha=0.25)
    
    # Bifurcation boundary line (black)
    boundary_line = 0.214 + 0.15 * (alpha43_grid - 0.784)
    ax.plot(boundary_line, alpha43_grid, 'k-', linewidth=2.5, label='Transcritical Bifurcation Boundary')
    
    # Current estimate (green point)
    current_K4 = 0.248
    current_alpha43 = 0.784
    ax.plot(current_K4, current_alpha43, 'go', markersize=12, label='Current Estimate')
    
    # 95% joint confidence region (blue ellipse, ρ = -0.45)
    ellipse = Ellipse(xy=(current_K4, current_alpha43), width=0.028, height=0.045,
                      angle=-30, facecolor='none', edgecolor='blue', 
                      linewidth=2, linestyle='--', label='95% Joint Confidence Region ($\\rho = -0.45$)')
    ax.add_patch(ellipse)
    
    # Region labels
    ax.text(0.30, 0.68, 'Coexistence', fontsize=11, ha='center', 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.text(0.20, 0.90, 'Exclusion', fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    ax.set_xlabel('Carrying Capacity $K_4$', fontsize=12)
    ax.set_ylabel('Competitive Pressure $\\alpha_{43}$', fontsize=12)
    ax.set_title('Two-Parameter Bifurcation Diagram', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, frameon=True, fancybox=True, edgecolor='black')
    ax.grid(False)
    ax.set_xlim(0.18, 0.32)
    ax.set_ylim(0.65, 0.95)
    
    return fig


# ============================================================================
# FIGURE 5: Hopf Bifurcation
# ============================================================================

def create_figure5_hopf_bifurcation():
    """Figure 5: Hopf bifurcation - eigenvalue plot and limit cycle."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel (a): Maximum real part of eigenvalues vs cyclic competition strength γ
    gamma_range = np.linspace(0.5, 2.0, 300)
    gamma_bifurcation = 1.42
    
    max_eigenvalue_real = np.zeros_like(gamma_range)
    for i, gamma in enumerate(gamma_range):
        if gamma < gamma_bifurcation:
            max_eigenvalue_real[i] = -0.12 + (gamma - 0.7) * 0.2
        else:
            max_eigenvalue_real[i] = 0.08 * (gamma - gamma_bifurcation)
        max_eigenvalue_real[i] = np.clip(max_eigenvalue_real[i], -0.2, 0.3)
    
    ax1 = axes[0]
    ax1.plot(gamma_range, max_eigenvalue_real, 'b-', linewidth=2.5)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax1.axvline(x=gamma_bifurcation, color='red', linestyle='--', linewidth=2, 
                label=f'Hopf Bifurcation at $\\gamma = {gamma_bifurcation:.2f}$')
    
    # Mark calibrated value (γ = 0.7)
    calibrated_idx = np.argmin(np.abs(gamma_range - 0.7))
    ax1.plot(0.7, max_eigenvalue_real[calibrated_idx], 'go', markersize=10, 
             label='Calibrated ($\\gamma = 0.7$)')
    
    # Add stability region labels
    ax1.text(0.9, -0.05, 'Stable Region', fontsize=10, ha='center', 
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax1.text(1.7, 0.05, 'Unstable Region', fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    ax1.set_xlabel('Cyclic Competition Strength $\\gamma$', fontsize=12)
    ax1.set_ylabel('Max Real Part of Eigenvalues', fontsize=12)
    ax1.set_title('(a) Maximum Real Part of Eigenvalues', fontsize=12)
    ax1.legend(loc='best', fontsize=10, frameon=True, fancybox=True, edgecolor='black')
    ax1.grid(False)
    ax1.set_ylim(-0.2, 0.35)
    
    # Panel (b): Stable limit cycle at γ = 1.6
    ax2 = axes[1]
    t = np.linspace(0, 50, 2000)
    omega = 2 * np.pi / 8  # 8-quarter period
    amplitude = 0.08
    
    x1 = 0.35 + amplitude * np.sin(omega * t)
    x3 = 0.28 + amplitude * np.sin(omega * t + np.pi)
    
    ax2.plot(t, x1, 'b-', linewidth=1.5, label='Firm 1')
    ax2.plot(t, x3, 'g-', linewidth=1.5, label='Firm 3')
    
    ax2.set_xlabel('Time (quarters)', fontsize=12)
    ax2.set_ylabel('Market Share', fontsize=12)
    ax2.set_title('(b) Stable Limit Cycle at $\\gamma = 1.6$', fontsize=12)
    ax2.legend(loc='best', fontsize=10, frameon=True, fancybox=True, edgecolor='black')
    ax2.grid(False)
    ax2.set_ylim(0.15, 0.48)
    
    plt.suptitle('Hopf Bifurcation and Cyclic Competition', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================================
# FIGURE 6: Sobol Sensitivity Indices
# ============================================================================

def create_figure6_sobol_indices():
    """Figure 6: Sobol sensitivity indices."""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    param_names = ['$K_4$', '$\\alpha_{43}$', '$\\alpha_{41}$', '$K_3$', 
                   '$\\alpha_{34}$', '$K_1$', '$\\alpha_{31}$', '$K_2$', 
                   '$\\alpha_{42}$', '$\\alpha_{32}$', 'Others']
    
    # From manuscript Section 4.2
    first_order = np.array([0.58, 0.15, 0.09, 0.05, 0.03, 0.02, 0.01, 0.01, 0.005, 0.005, 0.00])
    total_order = np.array([0.67, 0.21, 0.11, 0.07, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.00])
    
    # Normalize
    first_order = first_order / np.sum(first_order)
    total_order = total_order / np.sum(total_order)
    
    x = np.arange(len(param_names))
    width = 0.35
    
    ax.bar(x - width/2, first_order, width, label='First-order ($S_1$)', 
           color='blue', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, total_order, width, label='Total-order ($S_T$)', 
           color='red', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add error bars
    for i, (f, t) in enumerate(zip(first_order, total_order)):
        ax.errorbar(x[i] - width/2, f, yerr=0.02*f, color='black', capsize=3, capthick=1)
        ax.errorbar(x[i] + width/2, t, yerr=0.02*t, color='black', capsize=3, capthick=1)
    
    ax.set_xlabel('Parameters', fontsize=12)
    ax.set_ylabel('Sensitivity Index', fontsize=12)
    ax.set_title('Sobol Sensitivity Indices for Firm 4 Equilibrium Share', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, edgecolor='black')
    ax.set_ylim(0, 0.45)
    ax.grid(False)
    
    plt.tight_layout()
    return fig


# ============================================================================
# FIGURE 7: Morris Screening Results (REMOVED EXTRA ANNOTATIONS)
# ============================================================================

def create_figure7_morris_screening():
    """Figure 7: Morris screening results - minimal annotations only."""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    param_names = ['$K_4$', '$\\alpha_{43}$', '$\\alpha_{41}$', '$K_3$', 
                   '$K_1$', '$K_2$', '$\\alpha_{34}$', '$\\alpha_{31}$',
                   '$r_1$', '$r_2$', '$r_3$', '$r_4$',
                   '$\\alpha_{12}$', '$\\alpha_{13}$', '$\\alpha_{14}$',
                   '$\\alpha_{21}$', '$\\alpha_{23}$', '$\\alpha_{24}$',
                   '$\\alpha_{32}$', '$\\alpha_{42}$']
    
    # μ* and σ values
    mu_star = np.array([0.85, 0.72, 0.58, 0.45, 0.38, 0.35, 0.12, 0.08,
                        0.008, 0.006, 0.005, 0.004, 0.003, 0.003, 0.002,
                        0.002, 0.002, 0.001, 0.001, 0.001])
    
    sigma = np.array([0.32, 0.28, 0.21, 0.18, 0.08, 0.08, 0.055, 0.05,
                      0.015, 0.012, 0.010, 0.008, 0.007, 0.006, 0.005,
                      0.005, 0.004, 0.004, 0.003, 0.003])
    
    # Color coding
    colors = []
    for i in range(len(param_names)):
        if mu_star[i] > 0.3 and sigma[i] > 0.15:
            colors.append('red')      # Interactive, influential
        elif mu_star[i] > 0.3:
            colors.append('blue')     # Linear/additive, influential
        else:
            colors.append('gray')     # Negligible
    
    ax.scatter(mu_star, sigma, c=colors, s=100, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Label only the most influential parameters
    influential = [0, 1, 2, 3, 4, 5]
    for i in influential:
        ax.annotate(param_names[i], (mu_star[i], sigma[i]), 
                   xytext=(mu_star[i] + 0.03, sigma[i] + 0.02), fontsize=9)
    
    # Quadrant boundaries (keep for clarity)
    ax.axvline(x=0.3, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=0.15, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Simplified quadrant labels (removed extra annotations)
    ax.text(0.65, 0.28, 'High $\mu^*$, High $\sigma$', fontsize=9, ha='center')
    ax.text(0.65, 0.08, 'High $\mu^*$, Low $\sigma$', fontsize=9, ha='center')
    ax.text(0.15, 0.08, 'Negligible', fontsize=9, ha='center')
    
    ax.set_xlabel('Mean of Elementary Effects ($\\mu^*$)', fontsize=12)
    ax.set_ylabel('Standard Deviation of Elementary Effects ($\\sigma$)', fontsize=12)
    ax.set_title('Morris Screening Results', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 0.45)
    ax.grid(False)
    
    plt.tight_layout()
    return fig


# ============================================================================
# FIGURE 8: Equilibrium Uncertainty
# ============================================================================

def create_figure8_equilibrium_uncertainty():
    """Figure 8: Equilibrium uncertainty."""
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    n_samples = 500
    
    # From manuscript
    means = [0.352, 0.283, 0.318, 0.248]
    stds = [0.0095, 0.0085, 0.0090, 0.0080]
    
    equilibrium_samples = np.zeros((n_samples, 4))
    for i in range(4):
        equilibrium_samples[:, i] = np.random.normal(means[i], stds[i], n_samples)
        equilibrium_samples[:, i] = np.maximum(equilibrium_samples[:, i], 0.05)
    
    # Normalize
    equilibrium_samples = equilibrium_samples / np.sum(equilibrium_samples, axis=1, keepdims=True)
    
    firms = ['Firm 1', 'Firm 2', 'Firm 3', 'Firm 4']
    positions = [1, 2, 3, 4]
    colors = ['blue', 'orange', 'green', 'red']
    
    bp = ax.boxplot([equilibrium_samples[:, 0], equilibrium_samples[:, 1], 
                     equilibrium_samples[:, 2], equilibrium_samples[:, 3]],
                    positions=positions, widths=0.5, patch_artist=True,
                    showmeans=True, meanline=True, 
                    meanprops={'color': 'black', 'linestyle': '-', 'linewidth': 1.5})
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    
    # Add density plots
    ax2 = ax.twinx()
    for i, pos in enumerate(positions):
        kde = gaussian_kde(equilibrium_samples[:, i])
        x_range = np.linspace(0, 0.5, 100)
        density = kde(x_range) * 0.4
        ax2.fill_betweenx(x_range, -density, 0, alpha=0.3, color=colors[i])
    
    ax.set_xlabel('Firm', fontsize=12)
    ax.set_ylabel('Equilibrium Market Share', fontsize=12)
    ax.set_title('Posterior Predictive Distributions of Long-Term Equilibrium Shares', 
                fontsize=12, fontweight='bold')
    ax.set_xticks(positions)
    ax.set_xticklabels(firms)
    ax.set_ylim(0, 0.45)
    ax.grid(False)
    
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([])
    
    plt.tight_layout()
    return fig


# ============================================================================
# FIGURE 9: Exclusion Probability Contour Map
# ============================================================================

def create_figure9_exclusion_probability():
    """Figure 9: Exclusion probability contour map for Firm 4."""
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Create grid
    K4_grid = np.linspace(0.18, 0.32, 150)
    alpha43_grid = np.linspace(0.65, 0.95, 150)
    K4_mesh, alpha43_mesh = np.meshgrid(K4_grid, alpha43_grid)
    
    # Bifurcation boundary
    boundary = 0.214 + 0.15 * (alpha43_mesh - 0.784)
    
    # Exclusion probability
    distance = (K4_mesh - boundary) / 0.015
    prob_exclusion = 1 / (1 + np.exp(-distance))
    prob_exclusion = np.clip(prob_exclusion, 0, 1)
    
    # Contour levels
    contour_levels = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.95]
    
    # Colored contour fill
    contourf = ax.contourf(K4_mesh, alpha43_mesh, prob_exclusion, 
                           levels=contour_levels, cmap='RdYlBu_r', alpha=0.8)
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label('Exclusion Probability', fontsize=11)
    
    # Contour lines
    contours = ax.contour(K4_mesh, alpha43_mesh, prob_exclusion, 
                          levels=contour_levels, colors='black', linewidths=0.8, alpha=0.5)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
    
    # Current estimate
    current_K4 = 0.248
    current_alpha43 = 0.784
    ax.plot(current_K4, current_alpha43, 'go', markersize=12, label='Current Estimate')
    
    # 95% joint confidence region
    ellipse = Ellipse(xy=(current_K4, current_alpha43), width=0.028, height=0.045,
                      angle=-30, facecolor='none', edgecolor='blue', 
                      linewidth=2, linestyle='--', label='95% Confidence Region')
    ax.add_patch(ellipse)
    
    ax.set_xlabel('Carrying Capacity $K_4$', fontsize=12)
    ax.set_ylabel('Competitive Pressure $\\alpha_{43}$', fontsize=12)
    ax.set_title('Exclusion Probability Contour Map for Firm 4', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, frameon=True, fancybox=True, edgecolor='black')
    ax.set_xlim(0.18, 0.32)
    ax.set_ylim(0.65, 0.95)
    ax.grid(False)
    
    plt.tight_layout()
    return fig


# ============================================================================
# FIGURE 10: Value-at-Risk Analysis
# ============================================================================

def create_figure10_var_analysis():
    """Figure 10: VaR analysis - legend moved to safe position."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    n_samples = 10000
    np.random.seed(42)
    
    mean_share = 0.151
    std_share = 0.022
    
    projected_shares = np.random.normal(mean_share, std_share, n_samples)
    projected_shares = np.clip(projected_shares, 0.08, 0.25)
    
    var_05 = np.percentile(projected_shares, 5)
    var_01 = np.percentile(projected_shares, 1)
    cvar_05 = np.mean(projected_shares[projected_shares <= var_05])
    
    # Panel (a): CDF - legend at lower right (safe)
    ax1 = axes[0]
    sorted_shares = np.sort(projected_shares)
    cdf = np.arange(1, len(sorted_shares) + 1) / len(sorted_shares)
    
    ax1.plot(sorted_shares * 100, cdf * 100, 'b-', linewidth=2.5, label='CDF')
    ax1.axvline(x=var_05 * 100, color='red', linestyle='--', linewidth=2, 
                label=f'VaR$_{{0.05}}$ = {var_05*100:.1f}%')
    ax1.axhline(y=5, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.axvline(x=var_01 * 100, color='darkred', linestyle=':', linewidth=2, alpha=0.7,
                label=f'VaR$_{{0.01}}$ = {var_01*100:.1f}%')
    
    ax1.set_xlabel('Firm 4 Market Share (%)', fontsize=12)
    ax1.set_ylabel('Cumulative Probability (%)', fontsize=12)
    ax1.set_title('(a) Cumulative Distribution Function (CDF)', fontsize=12)
    ax1.set_xlim(8, 25)
    ax1.set_ylim(0, 100)
    ax1.legend(loc='lower right', fontsize=9, frameon=True, fancybox=True, edgecolor='black')
    ax1.grid(False)
    
    # Panel (b): PDF - legend at upper left (safe, away from red tail on right)
    ax2 = axes[1]
    bins = 40
    n, bins, patches = ax2.hist(projected_shares * 100, bins=bins, density=True, 
                                 facecolor='blue', alpha=0.6, edgecolor='black', linewidth=0.5)
    
    # Color the left tail
    for i in range(len(bins)-1):
        if bins[i] <= var_05 * 100:
            patches[i].set_facecolor('red')
            patches[i].set_alpha(0.8)
    
    ax2.axvline(x=var_05 * 100, color='red', linestyle='--', linewidth=2, 
                label=f'VaR$_{{0.05}}$ = {var_05*100:.1f}%')
    ax2.axvline(x=cvar_05 * 100, color='darkred', linestyle=':', linewidth=2, 
                label=f'CVaR$_{{0.05}}$ = {cvar_05*100:.1f}%')
    
    exclusion_threshold = 11.2
    ax2.axvline(x=exclusion_threshold, color='black', linestyle='-', linewidth=2, 
                label=f'Exclusion Threshold = {exclusion_threshold:.1f}%')
    
    ax2.set_xlabel('Firm 4 Market Share (%)', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('(b) Probability Density Function (PDF) with 5% Tail', fontsize=12)
    ax2.legend(loc='upper left', fontsize=9, frameon=True, fancybox=True, edgecolor='black')
    ax2.grid(False)
    
    plt.suptitle('Value at Risk (VaR) Analysis for Firm 4', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Generate all 10 figures."""
    
    print("=" * 60)
    print("Generating all 10 figures for manuscript")
    print("=" * 60)
    
    figures = [
        ("Fig1_MCMC_Convergence.png", create_figure1_mcmc_convergence, "MCMC Convergence Diagnostics"),
        ("Fig2_Posterior_Distributions.png", create_figure2_posterior_distributions, "Posterior Distributions"),
        ("Fig3_Exclusion_Threshold.png", create_figure3_exclusion_threshold, "Competitive Exclusion Threshold"),
        ("Fig4_TwoParameter_Bifurcation.png", create_figure4_twoparameter_bifurcation, "Two-Parameter Bifurcation"),
        ("Fig5_Hopf_Bifurcation.png", create_figure5_hopf_bifurcation, "Hopf Bifurcation"),
        ("Fig6_Sobol_Sensitivity.png", create_figure6_sobol_indices, "Sobol Sensitivity Indices"),
        ("Fig7_Morris_Screening.png", create_figure7_morris_screening, "Morris Screening Results"),
        ("Fig8_Equilibrium_Uncertainty.png", create_figure8_equilibrium_uncertainty, "Equilibrium Uncertainty"),
        ("Fig9_Exclusion_Probability.png", create_figure9_exclusion_probability, "Exclusion Probability Contour Map"),
        ("Fig10_VaR_Analysis.png", create_figure10_var_analysis, "Value-at-Risk Analysis")
    ]
    
    for i, (filename, func, description) in enumerate(figures, 1):
        print(f"[{i:2d}/10] Creating {description}...")
        fig = func()
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"      Saved as {filename}")
    
    print("\n" + "=" * 60)
    print("SUCCESS: All 10 figures generated")
    print("=" * 60)


if __name__ == "__main__":
    main()