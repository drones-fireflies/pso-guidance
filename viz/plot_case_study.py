"""
plot_case_study.py
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from guidance_pso.firemodel.propagation import Environment, FirePropagation
from guidance_pso.guidance.guidance_controller import WildfireGuidanceController
from guidance_pso.perception.front_detection import frontier_cells, cv_frontier


# ── colour helpers ────────────────────────────────────────────────────────────

def _fire_rgb(arr):
    arr = np.clip(arr, 0.0, 1.0)
    rgb = np.zeros((*arr.shape, 3), dtype=np.float32)
    rgb[...,0] = np.where(arr>0, np.clip(arr*1.8+0.2,0,1), 0)
    rgb[...,1] = np.where(arr>0, np.clip(arr*0.5-0.05,0,1), 0)
    return rgb

def _belief_rgb(arr):
    arr = np.clip(arr, 0.0, 1.0)
    rgb = np.full((*arr.shape,3), [0.04,0.05,0.10], dtype=np.float32)
    m = arr > 0
    rgb[m,0] = 0.0
    rgb[m,1] = np.clip(arr[m]*0.75, 0, 1)
    rgb[m,2] = np.clip(arr[m]*1.0+0.2, 0, 1)
    return rgb

def _overlay_frontier(ax, front, color, s=1.2):
    pts = np.argwhere(front > 0)
    if len(pts):
        ax.scatter(pts[:,1], pts[:,0], s=s, c=color,
                   alpha=0.9, linewidths=0, zorder=4)

def _draw_quadcopter(ax, r, c, theta, col, size=2.5, alpha=0.95):
    arm=size; body=size*0.28; rotor=size*0.32
    ct,st = np.cos(theta), np.sin(theta)
    def rot(x,y): return (c+x*ct+y*st, r-x*st+y*ct)
    bpts = [rot(body,body),rot(-body,body),rot(-body,-body),rot(body,-body)]
    ax.add_patch(plt.Polygon(bpts,closed=True,
        fc=col,ec='white',lw=0.5,alpha=alpha,zorder=7))
    for ax_,ay_ in [(arm,0),(-arm,0),(0,arm),(0,-arm)]:
        x0,y0=rot(0,0); x1,y1=rot(ax_,ay_)
        ax.plot([x0,x1],[y0,y1],'-',color=col,lw=0.8,alpha=alpha,zorder=6)
        ax.add_patch(plt.Circle((x1,y1),rotor,
            fc='none',ec=col,lw=0.5,alpha=alpha*0.6,zorder=6))
    nose=[rot(size*1.1,0),rot(size*0.65,size*0.2),rot(size*0.65,-size*0.2)]
    ax.add_patch(plt.Polygon(nose,closed=True,
        fc='white',ec='none',alpha=alpha*0.9,zorder=8))

def _style(ax, H, W):
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0,W); ax.set_ylim(H,0)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6); sp.set_edgecolor('#666666')


# ── main generator ────────────────────────────────────────────────────────────

def generate_case_study(snapshots, n_drones, fov_half,
                         save_path='case_study.png', dpi=220):
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.serif':  ['Times New Roman','DejaVu Serif'],
        'font.size':   11,
    })

    step_keys = sorted(snapshots.keys())
    n_cols    = len(step_keys)
    colors    = plt.cm.gist_rainbow(np.linspace(0.05, 0.95, n_drones))

    cell_w  = 2.8    # inches per column — larger
    cell_h  = 2.8    # square cells
    label_w = 1.3    # wider left margin for bigger labels

    fig_w = label_w + n_cols * cell_w + 0.15
    fig_h = 3 * cell_h + 0.75

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')

    l = label_w / fig_w
    r = 1.0 - 0.05/fig_w
    t = 1.0 - 0.38/fig_h
    b = 0.70/fig_h

    gs = gridspec.GridSpec(3, n_cols,
        left=l, right=r, top=t, bottom=b,
        hspace=0.08, wspace=0.10)

    axes = [[fig.add_subplot(gs[ri, ci]) for ci in range(n_cols)]
            for ri in range(3)]

    f0 = snapshots[step_keys[0]]
    H, W = f0['true_fire'].shape

    for ci, sk in enumerate(step_keys):
        f = snapshots[sk]
        phase     = f['phase']
        slabel    = f.get('step_label', sk+1)
        phase_col = '#FF8800' if phase=='navigation' else '#00B8D9'
        phase_str = 'Nav.'    if phase=='navigation' else 'PSO'

        # ── row 0: true fire + frontier ──────────────────────────────────
        ax = axes[0][ci]
        ax.imshow(_fire_rgb(f['true_fire']), origin='upper',
                  interpolation='nearest', extent=[0,W,H,0],
                  aspect='equal')
        _overlay_frontier(ax, f['true_front'], '#FFD600', s=2.5)
        _style(ax, H, W)
        for sp in ax.spines.values():
            sp.set_edgecolor(phase_col); sp.set_linewidth(2.5)
        ax.set_title(f'Step {slabel}]',
                     fontsize=12, pad=4, color='#111111', fontweight='bold')

        # ── row 1: drone positions on white ──────────────────────────────
        ax = axes[1][ci]
        ax.set_facecolor('white')
        ax.set_aspect('equal')
        _style(ax, H, W)
        for sp in ax.spines.values():
            sp.set_edgecolor('#666666'); sp.set_linewidth(0.8)
        dp = f['drone_pos']
        ps = f['poses']
        for i in range(n_drones):
            r_, c_ = float(dp[i,0]), float(dp[i,1])
            theta  = float(ps[i,2])
            _draw_quadcopter(ax, r_, c_, theta, colors[i], size=4.0)
            ax.add_patch(plt.Rectangle(
                (c_-fov_half, r_-fov_half), 2*fov_half, 2*fov_half,
                lw=1.0, ec=colors[i], fc=(*colors[i][:3], 0.06),
                ls='--', zorder=5))

        # ── row 2: belief map + frontier ──────────────────────────────────
        ax = axes[2][ci]
        ax.imshow(_belief_rgb(f['belief_fire']), origin='upper',
                  interpolation='nearest', extent=[0,W,H,0],
                  aspect='equal')
        _overlay_frontier(ax, f['belief_front'], '#00E5FF', s=2.5)
        _style(ax, H, W)

    # ── row labels ────────────────────────────────────────────────────────────
    row_labels = [
        r'$\mathbf{X}_t \oplus \mathbf{R}_t$',
        r'$\{\mathbf{p}^{(i)}_t\}_{i=1}^{M}$',
        r'$\hat{\mathbf{X}}_t \oplus \hat{\mathbf{R}}_t$',
    ]
    for ri, lbl in enumerate(row_labels):
        pos = axes[ri][0].get_position()
        fig.text(
            l - 0.015, (pos.y0+pos.y1)/2, lbl,
            ha='right', va='center', fontsize=13,
            fontweight='bold', transform=fig.transFigure,
        )

    # ── legend ────────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(color='#e84a00', label=r'Fire $\mathbf{X}_t$'),
        mpatches.Patch(color='#FFD600', label=r'True frontier $\mathbf{R}_t$'),
        mpatches.Patch(color='#00E5FF', label=r'Belief frontier $\hat{\mathbf{R}}_t$'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3,
               fontsize=12, frameon=True, framealpha=0.95,
               edgecolor='#aaaaaa', bbox_to_anchor=(0.5, 0.0),
               handlelength=1.6, columnspacing=2.5, handletextpad=0.8)

    fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                facecolor='white',
                format='pdf' if save_path.endswith('.pdf') else 'png')
    print(f'Case study saved → {save_path}')
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


# ── standalone entry point ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps',       type=int, default=600)
    parser.add_argument('--drones',      type=int, default=5)
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--save',        type=str, default='case_study.png')
    parser.add_argument('--dpi',         type=int, default=220)
    parser.add_argument('--fire-period', type=int, default=10)
    parser.add_argument('--n-cols',      type=int, default=5,
                        help='Number of snapshot columns (default 5)')
    args = parser.parse_args()

    T        = args.steps
    M        = args.drones
    fire_per = args.fire_period
    n_cols   = args.n_cols

    # evenly spaced snapshots
    snap_at = set(np.linspace(0, T-1, n_cols, dtype=int))

    np.random.seed(args.seed)
    env      = Environment(grid_size=200, ignition_points=[(100,100)],
                           wind_direction=2.55)
    fire_sim = FirePropagation(timesteps=20, visualize=False)
    fire_states, fuel_states, _, _ = fire_sim.propagate_from_points(env)
    current_fire = fire_states[-1].copy()
    current_fuel = fuel_states[-1].copy()
    slope_effect = env._compute_slope_effect(env.elevation_map)

    ctrl = WildfireGuidanceController(
        fire_map=current_fire, n_drones=M,
        pso_particles=40, pso_iterations=30)

    snapshots = {}
    for t in range(T):
        m = ctrl.step()
        if (t+1) % fire_per == 0:
            current_fire, current_fuel = fire_sim._spread_fire(
                current_fire, current_fuel, slope_effect,
                env.wind_direction, env.wind_strength,
                env.burn_rate_map, env.moisture_map)
            ctrl.true_fire     = current_fire.copy()
            ctrl.belief.front_pts = frontier_cells(
                (current_fire*255).astype('uint8'), 60)
        if t in snap_at:
            tf = cv_frontier((current_fire*255).astype('uint8'), 60)
            snapshots[t] = dict(
                true_fire   = current_fire.copy(),
                true_front  = tf.copy(),
                belief_fire = ctrl.belief.fire_belief.copy(),
                belief_front= ctrl.belief.frontier.copy(),
                drone_pos   = ctrl.drones.copy(),
                poses       = ctrl.poses.copy(),
                phase       = m.phase,
                step_label  = t+1,
            )
            print(f'  snapshot t={t+1:4d}  [{m.phase}]')

    generate_case_study(
        snapshots = snapshots,
        n_drones  = M,
        fov_half  = ctrl.fov_half,
        save_path = args.save,
        dpi       = args.dpi,
    )

if __name__ == '__main__':
    main()
