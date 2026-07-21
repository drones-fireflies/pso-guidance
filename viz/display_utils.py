"""
display_utils.py
================
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as anim_mod
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from scipy.interpolate import make_interp_spline

from typing import List
from guidance_pso.metrics import CoverageMetrics


def _fire_rgb(fire: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*fire.shape, 3), dtype=np.float32)
    rgb[fire > 0.5]                 = [1.00, 0.35, 0.00]
    rgb[(fire > 0) & (fire <= 0.5)] = [0.55, 0.12, 0.00]
    return rgb


def _overlay_frontier(ax, frontier: np.ndarray, color: str, s: float = 0.8):
    pts = np.argwhere(frontier > 0)
    if len(pts):
        ax.scatter(pts[:, 1], pts[:, 0], s=s, c=color, alpha=0.7,
                   linewidths=0, zorder=4)


def _uav_polygon(cx, cy, theta, size, col, alpha=0.95):
    arm = size * 1.0
    body = size * 0.28
    rotor = size * 0.32
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    def _rot(x, y):
        return (cx + x*cos_t + y*sin_t, cy - x*sin_t + y*cos_t)

    patches = []
    bpts = [_rot(body, body), _rot(-body, body),
            _rot(-body, -body), _rot(body, -body)]
    patches.append(matplotlib.patches.Polygon(
        bpts, closed=True, facecolor=col, edgecolor="white",
        linewidth=0.4, alpha=alpha, zorder=7))
    for (ax_, ay_) in [(arm, 0), (-arm, 0), (0, arm), (0, -arm)]:
        x0, y0 = _rot(0, 0)
        x1, y1 = _rot(ax_, ay_)
        patches.append(matplotlib.patches.FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle='-', color=col,
            linewidth=max(0.4, size * 0.35), alpha=alpha, zorder=6))
        patches.append(matplotlib.patches.Circle(
            (x1, y1), rotor, facecolor="none", edgecolor=col,
            linewidth=max(0.4, size * 0.25), alpha=alpha * 0.6, zorder=6))
    nose_pts = [_rot(size*1.15, 0), _rot(size*0.7, size*0.18),
                _rot(size*0.7, -size*0.18)]
    patches.append(matplotlib.patches.Polygon(
        nose_pts, closed=True, facecolor="white", edgecolor="none",
        alpha=alpha * 0.9, zorder=8))
    return patches


def _draw_drones(ax, drone_pos, fov_half, colors, poses=None,
                 size=None, collision_pairs=None):
    if drone_pos is None:
        return
    if size is None:
        size = 1.0
    if collision_pairs is None:
        collision_pairs = set()
    colliding = set()
    for i, j in collision_pairs:
        colliding.add(i); colliding.add(j)
    for i, (r, c) in enumerate(drone_pos.astype(float)):
        col = colors[i]
        theta = float(poses[i, 2]) if poses is not None and i < len(poses) else 0.0
        for patch in _uav_polygon(c, r, theta, size, col):
            ax.add_patch(patch)
        if i in colliding:
            ax.add_patch(matplotlib.patches.Circle(
                (c, r), size * 2.2, facecolor="none", edgecolor="#ff0000",
                linewidth=1.5, alpha=0.9, zorder=9, linestyle="--"))
        ax.add_patch(plt.Rectangle(
            (c - fov_half, r - fov_half), 2 * fov_half, 2 * fov_half,
            linewidth=0.4, edgecolor=col, facecolor="none",
            alpha=0.25, linestyle="--", zorder=5))


def _smooth_trail(xs, ys, n_pts=200):
    xs, ys = np.array(xs, dtype=float), np.array(ys, dtype=float)
    n = len(xs)
    if n < 4:
        return xs, ys
    dist = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    keep = np.concatenate([[True], dist > 0.01])
    xs, ys = xs[keep], ys[keep]
    if len(xs) < 4:
        return xs, ys
    t = np.linspace(0, 1, len(xs))
    t_fine = np.linspace(0, 1, n_pts)
    try:
        spl_x = make_interp_spline(t, xs, k=3)
        spl_y = make_interp_spline(t, ys, k=3)
        return spl_x(t_fine), spl_y(t_fine)
    except Exception:
        return xs, ys


def _style_cell(ax, title="", fontsize=7):
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5); spine.set_edgecolor("#aaaaaa")
    if title:
        ax.set_title(title, fontsize=fontsize, fontweight="bold",
                     color="#222222", pad=2)


_BG             = "#0d0d0f"
_PANEL_BG       = "#13141a"
_GRID_COL       = "#1e2030"
_TEXT           = "#dce1f0"
_SUBTEXT        = "#7a82a0"
_ACCENT         = "#4fc3f7"
_GOLD           = "#ffd54f"
_TRAJ_ALPHA_MAX = 0.75
_TRAJ_ALPHA_MIN = 0.05
_TRAJ_LW        = 1.1


def _fire_rgb_hd(fire):
    fire = np.clip(fire, 0.0, 1.0)
    rgb = np.zeros((*fire.shape, 3), dtype=np.float32)
    rgb[..., 0] = np.where(fire > 0, np.clip(fire * 1.8 + 0.2, 0, 1), 0)
    rgb[..., 1] = np.where(fire > 0, np.clip(fire * 0.55 - 0.05, 0, 1), 0)
    rgb[..., 2] = 0.0
    return rgb


def _belief_rgb(fire):
    H, W = fire.shape
    fire = np.clip(fire, 0.0, 1.0)
    rgb  = np.full((H, W, 3), [0.05, 0.06, 0.12], dtype=np.float32)
    mask = fire > 0
    rgb[mask, 0] = 0.0
    rgb[mask, 1] = np.clip(fire[mask] * 0.8, 0, 1)
    rgb[mask, 2] = np.clip(fire[mask] * 1.0 + 0.15, 0, 1)
    return rgb


def _panel_spine(ax, color="#2a2d3e", lw=0.8):
    for sp in ax.spines.values():
        sp.set_edgecolor(color); sp.set_linewidth(lw)


def _panel_title(ax, title, color=_TEXT):
    ax.set_title(title, color=color, fontsize=8.5, pad=5,
                 fontfamily="monospace", fontweight="bold", loc="left")


class RealtimeAnimator:
    _TRAIL_LEN = 60

    def __init__(self, n_drones, fov_half, grid_shape):
        self.N        = n_drones
        self.fov_half = fov_half
        self.H, self.W = grid_shape
        self._frames  = []
        self._trails  = [[] for _ in range(n_drones)]
        self.colors   = plt.cm.gist_rainbow(np.linspace(0.05, 0.95, n_drones))

        self.fig = plt.figure(figsize=(15, 5.2), facecolor=_BG)
        gs = gridspec.GridSpec(1, 3, figure=self.fig,
                               left=0.02, right=0.98,
                               bottom=0.10, top=0.88, wspace=0.04)
        self.ax_fire = self.fig.add_subplot(gs[0])
        self.ax_bel  = self.fig.add_subplot(gs[1])
        self.ax_traj = self.fig.add_subplot(gs[2])

        for ax in (self.ax_fire, self.ax_bel, self.ax_traj):
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor(_PANEL_BG)
            _panel_spine(ax)

        _panel_title(self.ax_fire, "[1] True propagation")
        _panel_title(self.ax_bel,  "[2] belief propagation")
        _panel_title(self.ax_traj, "[3] UAVs real-time mission")

        blank_rgb = np.zeros((self.H, self.W, 3), dtype=np.float32)
        blank_wh  = np.ones( (self.H, self.W, 3), dtype=np.float32)

        self._img_fire = self.ax_fire.imshow(
            blank_rgb, origin="upper", interpolation="nearest",
            extent=[0, self.W, self.H, 0])
        self._img_bel  = self.ax_bel.imshow(
            blank_rgb, origin="upper", interpolation="nearest",
            extent=[0, self.W, self.H, 0])
        self._img_traj = self.ax_traj.imshow(
            blank_wh, origin="upper", interpolation="nearest",
            extent=[0, self.W, self.H, 0], zorder=0)

        for ax in (self.ax_fire, self.ax_bel, self.ax_traj):
            ax.set_xlim(0, self.W); ax.set_ylim(self.H, 0)

        self._sc_tf = self.ax_fire.scatter(
            [], [], s=0.8, c=_GOLD,   alpha=0.75, linewidths=0, zorder=5)
        self._sc_bf = self.ax_bel.scatter(
            [], [], s=0.8, c=_ACCENT, alpha=0.75, linewidths=0, zorder=5)

        self._drone_patch_groups = []
        self._fov_rects          = []
        self._last_poses         = None
        self._arrow_len          = 1.0

        for i in range(n_drones):
            col = self.colors[i]
            fh  = self.fov_half
            placeholders = _uav_polygon(0, 0, 0.0, self._arrow_len, col, alpha=0.0)
            for p in placeholders:
                self.ax_traj.add_patch(p)
            self._drone_patch_groups.append(placeholders)
            rect = plt.Rectangle(
                (0, 0), 2*fh, 2*fh,
                linewidth=0.4, edgecolor=col,
                facecolor="none", alpha=0.0, zorder=5)
            self.ax_traj.add_patch(rect)
            self._fov_rects.append(rect)

        self._traj_lines = []
        for i in range(n_drones):
            line, = self.ax_traj.plot(
                [], [], lw=_TRAJ_LW + 0.4, color=self.colors[i],
                alpha=0.75, solid_capstyle="round",
                solid_joinstyle="round", zorder=3)
            self._traj_lines.append(line)

        self._collision_txt = self.ax_traj.text(
            0.98, 0.02, "", transform=self.ax_traj.transAxes,
            ha="right", va="bottom", fontsize=7.5,
            fontfamily="monospace", fontweight="bold",
            color="#ff4444", zorder=10)
        self._total_collisions = 0

        self.ax_fire.legend(
            handles=[mpatches.Patch(color="#e84a00", label="fire"),
                     mpatches.Patch(color=_GOLD,     label="frontier")],
            loc="lower left", fontsize=6.5,
            facecolor="#1a1b26", edgecolor="#2a2d3e",
            labelcolor=_TEXT, framealpha=0.85, handlelength=1.2)
        self.ax_bel.legend(
            handles=[mpatches.Patch(color="#00b8d9", label="belief fire"),
                     mpatches.Patch(color=_ACCENT,   label="belief frontier")],
            loc="lower left", fontsize=6.5,
            facecolor="#1a1b26", edgecolor="#2a2d3e",
            labelcolor=_TEXT, framealpha=0.85, handlelength=1.2)

        self._label_txt = self.fig.text(
            0.5, 0.015, "", ha="center", va="bottom",
            color=_SUBTEXT, fontsize=8.5, fontfamily="monospace")
        self._phase_txt = self.fig.text(
            0.98, 0.92, "", ha="right", va="top",
            color=_ACCENT, fontsize=9,
            fontfamily="monospace", fontweight="bold")

        plt.ion()
        self.fig.show()

    def push_frame(self, true_fire, true_frontier, belief_fire,
                   belief_frontier, drone_pos, label="",
                   uncertainty=None, poses=None):

        self._img_fire.set_data(_fire_rgb_hd(true_fire))
        tf = np.argwhere(true_frontier > 0)
        self._sc_tf.set_offsets(tf[:, ::-1] if len(tf) else np.empty((0, 2)))

        self._img_bel.set_data(_belief_rgb(belief_fire))
        bf = np.argwhere(belief_frontier > 0)
        self._sc_bf.set_offsets(bf[:, ::-1] if len(bf) else np.empty((0, 2)))

        if poses is not None:
            self._last_poses = poses

        DANGER_CELLS  = 1.0
        WARNING_CELLS = 10.0
        danger_pairs  = set()
        warning_pairs = set()

        if drone_pos is not None:
            for i in range(self.N):
                for j in range(i+1, self.N):
                    d = float(np.linalg.norm(
                        drone_pos[i].astype(float) - drone_pos[j].astype(float)))
                    if d < DANGER_CELLS:
                        danger_pairs.add((i, j))
                    elif d < WARNING_CELLS:
                        warning_pairs.add((i, j))

        if drone_pos is not None:
            for i in range(self.N):
                r, c = float(drone_pos[i, 0]), float(drone_pos[i, 1])
                self._trails[i].append((c, r))
                if len(self._trails[i]) > self._TRAIL_LEN:
                    self._trails[i].pop(0)

                trail = self._trails[i]
                if len(trail) > 1:
                    xs = [p[0] for p in trail]
                    ys = [p[1] for p in trail]
                    sx, sy = _smooth_trail(xs, ys, n_pts=max(len(trail)*6, 60))
                    n_pts_trail = len(sx)
                    alphas = np.linspace(_TRAJ_ALPHA_MIN, _TRAJ_ALPHA_MAX, n_pts_trail)
                    pts  = np.array([sx, sy]).T.reshape(-1, 1, 2)
                    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                    lc   = matplotlib.collections.LineCollection(
                        segs, linewidths=_TRAJ_LW + 0.4,
                        color=self.colors[i], alpha=None, zorder=3)
                    if len(alphas) > 1:
                        lc.set_alpha(alphas[:-1])
                    self._traj_lines[i].set_data([], [])
                    self.ax_traj.add_collection(lc)
                    if not hasattr(self, '_trail_collections'):
                        self._trail_collections = [None] * self.N
                    if self._trail_collections[i] is not None:
                        try: self._trail_collections[i].remove()
                        except Exception: pass
                    self._trail_collections[i] = lc

                for p in self._drone_patch_groups[i]:
                    try: p.remove()
                    except Exception: pass

                theta = float(self._last_poses[i, 2]) \
                    if self._last_poses is not None else 0.0
                col = self.colors[i]
                new_patches = _uav_polygon(c, r, theta, self._arrow_len, col)

                in_danger  = any(i in pair for pair in danger_pairs)
                in_warning = any(i in pair for pair in warning_pairs)
                if in_danger:
                    new_patches.append(matplotlib.patches.Circle(
                        (c, r), self._arrow_len * 2.5,
                        facecolor="#ff000033", edgecolor="#ff0000",
                        linewidth=2.0, alpha=0.95, zorder=9))
                elif in_warning:
                    new_patches.append(matplotlib.patches.Circle(
                        (c, r), self._arrow_len * 2.0,
                        facecolor="none", edgecolor="#ffaa00",
                        linewidth=1.2, alpha=0.75, zorder=9,
                        linestyle="--"))

                for p in new_patches:
                    self.ax_traj.add_patch(p)
                self._drone_patch_groups[i] = new_patches

                fh = self.fov_half
                self._fov_rects[i].set_xy((c - fh, r - fh))
                self._fov_rects[i].set_alpha(0.35)

        if danger_pairs:
            self._total_collisions += len(danger_pairs)
        if danger_pairs:
            pairs_str = " ".join(f"D{i}↔D{j}" for i,j in sorted(danger_pairs))
            self._collision_txt.set_text(
                f"⚠ DANGER <5m: {pairs_str}  (total={self._total_collisions})")
            self._collision_txt.set_color("#ff2222")
        elif warning_pairs:
            pairs_str = " ".join(f"D{i}↔D{j}" for i,j in sorted(warning_pairs))
            self._collision_txt.set_text(f"△ SEP <30m: {pairs_str}")
            self._collision_txt.set_color("#ffaa00")
        else:
            self._collision_txt.set_text(
                f"✓ safe  (collisions={self._total_collisions})")
            self._collision_txt.set_color("#69ff94")

        self._label_txt.set_text(label)
        phase = "PSO" if "pso" in label.lower() else "NAV"
        col   = _ACCENT if phase == "PSO" else _GOLD
        self._phase_txt.set_text(f"[ {phase} ]")
        self._phase_txt.set_color(col)

        self._frames.append(dict(
            true_fire       = true_fire.copy(),
            true_frontier   = true_frontier.copy(),
            belief_fire     = belief_fire.copy(),
            belief_frontier = belief_frontier.copy(),
            drone_pos       = drone_pos.copy() if drone_pos is not None else None,
            poses           = poses.copy() if poses is not None else None,
            trails          = [list(t) for t in self._trails],
            label           = label,
        ))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def finalize(self):
        self._label_txt.set_text("Simulation complete")
        self._phase_txt.set_text("[ DONE ]")
        self._phase_txt.set_color("#69ff94")
        self.fig.canvas.draw()
        plt.ioff()

    def save(self, path, fps=5, dpi=120):
        N  = self.N
        fh = self.fov_half
        colors = self.colors

        fig2 = plt.figure(figsize=(15, 5.2), facecolor=_BG)
        gs2  = gridspec.GridSpec(1, 3, figure=fig2,
                                 left=0.02, right=0.98,
                                 bottom=0.10, top=0.88, wspace=0.04)
        ax_f = fig2.add_subplot(gs2[0])
        ax_b = fig2.add_subplot(gs2[1])
        ax_t = fig2.add_subplot(gs2[2])

        for ax, ttl in [
            (ax_f, "[1] FIRE PROPAGATION  +  FRONT"),
            (ax_b, "[2] BELIEF MAP  +  BELIEF FRONT"),
            (ax_t, "[3] UAV TRAJECTORIES"),
        ]:
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor(_PANEL_BG)
            ax.set_xlim(0, self.W); ax.set_ylim(self.H, 0)
            _panel_spine(ax)
            _panel_title(ax, ttl)

        blank_rgb = np.zeros((self.H, self.W, 3), dtype=np.float32)
        blank_wh  = np.ones( (self.H, self.W, 3), dtype=np.float32)

        im_f = ax_f.imshow(blank_rgb, origin="upper", interpolation="nearest",
                           extent=[0,self.W,self.H,0])
        im_b = ax_b.imshow(blank_rgb, origin="upper", interpolation="nearest",
                           extent=[0,self.W,self.H,0])
        im_t = ax_t.imshow(blank_wh,  origin="upper", interpolation="nearest",
                           extent=[0,self.W,self.H,0], zorder=0)

        sc_tf = ax_f.scatter([], [], s=1.0, c=_GOLD,   alpha=0.8,
                             linewidths=0, zorder=5)
        sc_bf = ax_b.scatter([], [], s=1.0, c=_ACCENT, alpha=0.8,
                             linewidths=0, zorder=5)

        tlines, tdots, trects = [], [], []
        for i in range(N):
            col = colors[i]
            ln, = ax_t.plot([], [], lw=_TRAJ_LW + 0.4, color=col, alpha=0.75,
                            solid_capstyle="round", solid_joinstyle="round",
                            zorder=3)
            dt  = ax_t.scatter([], [], s=70, color=col, marker="^",
                               edgecolors="#222222", linewidths=0.6, zorder=6)
            rc  = plt.Rectangle((0,0), 2*fh, 2*fh, lw=0.8,
                                 edgecolor=col, facecolor=(*col[:3],0.06),
                                 zorder=4, visible=False)
            ax_t.add_patch(rc)
            tlines.append(ln); tdots.append(dt); trects.append(rc)

        lbl = fig2.text(0.5, 0.015, "", ha="center", va="bottom",
                        color=_SUBTEXT, fontsize=8.5, fontfamily="monospace")
        ph  = fig2.text(0.98, 0.92, "", ha="right", va="top",
                        color=_ACCENT, fontsize=9,
                        fontfamily="monospace", fontweight="bold")

        artists = [im_f, im_b, im_t, sc_tf, sc_bf, lbl, ph,
                   *tlines, *tdots, *trects]

        def _upd(frame_idx):
            f = self._frames[frame_idx]
            im_f.set_data(_fire_rgb_hd(f["true_fire"]))
            im_b.set_data(_belief_rgb(f["belief_fire"]))
            tf = np.argwhere(f["true_frontier"] > 0)
            sc_tf.set_offsets(tf[:,::-1] if len(tf) else np.empty((0,2)))
            bf = np.argwhere(f["belief_frontier"] > 0)
            sc_bf.set_offsets(bf[:,::-1] if len(bf) else np.empty((0,2)))
            trails = f["trails"]
            for i in range(N):
                trail = trails[i]
                if len(trail) > 1:
                    xs = [p[0] for p in trail]
                    ys = [p[1] for p in trail]
                    sx, sy = _smooth_trail(xs, ys, n_pts=max(len(trail)*6, 60))
                    tlines[i].set_data(sx, sy)
                dp = f["drone_pos"]
                if dp is not None:
                    c2, r2 = float(dp[i,1]), float(dp[i,0])
                    tdots[i].set_offsets([[c2, r2]])
                    trects[i].set_xy((c2-fh, r2-fh))
                    trects[i].set_visible(True)
            lbl.set_text(f["label"])
            ph.set_text("[ PSO ]" if "pso" in f["label"].lower() else "[ NAV ]")
            return artists

        ani = anim_mod.FuncAnimation(
            fig2, _upd, frames=len(self._frames), interval=200, blit=True)
        ani.save(path, writer="pillow", fps=fps, dpi=dpi)
        plt.close(fig2)
        print(f"GIF saved → {path}")

    def get_frame(self, index=-1):
        return self._frames[index]

    def get_all_frames(self):
        return self._frames


def plot_figure2(true_fire, true_frontier, belief_fire, belief_frontier,
                 drone_pos, fov_half, poses=None, step=0, phase="pso",
                 save_path=None):
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
        "figure.dpi": 150,
    })
    H = true_fire.shape[0]
    N = len(drone_pos)
    colors    = plt.cm.gist_rainbow(np.linspace(0.05, 0.95, N))
    arrow_len = max(true_fire.shape) * 0.035

    fig, (ax_true, ax_bel) = plt.subplots(1, 2, figsize=(7.16, 3.4),
                                           gridspec_kw={"wspace": 0.06})
    ax_true.imshow(_fire_rgb_hd(true_fire), origin="upper",
                   interpolation="nearest")
    _overlay_frontier(ax_true, true_frontier, color=_GOLD, s=1.2)
    _draw_drones(ax_true, drone_pos, fov_half, colors, poses=poses,
                 arrow_len=arrow_len)
    _style_cell(ax_true, title=r"(a) True fire state $F_t$  +  frontier")

    ax_bel.imshow(_belief_rgb(belief_fire), origin="upper",
                  interpolation="nearest")
    _overlay_frontier(ax_bel, belief_frontier, color=_ACCENT, s=1.2)
    _draw_drones(ax_bel, drone_pos, fov_half, colors, poses=poses,
                 arrow_len=arrow_len)
    _style_cell(ax_bel, title=r"(b) Belief map $\hat{F}_t$  +  frontier")

    fig.legend(
        handles=[mpatches.Patch(color="#e84a00", label="fire"),
                 mpatches.Patch(color=_GOLD,     label="true frontier"),
                 mpatches.Patch(color=_ACCENT,   label="belief frontier")],
        loc="lower center", ncol=3, fontsize=8, frameon=True, framealpha=0.9,
        edgecolor="#cccccc", bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Step {step}  —  phase: {phase}", fontsize=10, y=1.02,
                 fontfamily="serif")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight",
                    format="pdf" if save_path.endswith(".pdf") else "png")
        print(f"Figure 2 saved → {save_path}")
    plt.show()
    plt.close(fig)
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def _extract_step(label, fallback):
    import re
    m = re.search(r"step\s+(\d+)", label, re.IGNORECASE)
    if m: return f"Step {m.group(1)}"
    m = re.search(r"(\d+)", label)
    if m: return f"Step {m.group(1)}"
    return f"Step {fallback}"


def plot_metrics(ctrl, history: List[CoverageMetrics], save_path=None):
    """
    Four-panel metric figure — no Nav/PSO phase shading.
    Latency is shown for ALL steps (nav + pso).
    """
    if not history:
        print("No metric history to plot.")
        return

    plt.rcParams.update({
        "font.family":    "serif",
        "font.serif":     ["Times New Roman", "DejaVu Serif"],
        "font.size":      10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.8,
        "figure.dpi":     150,
    })

    steps = np.array([m.step for m in history])

    def _to_nan(arr):
        return np.where(arr >= 0.0, arr, np.nan)

    front_iou = _to_nan(np.array([m.frontier_iou        for m in history], dtype=np.float64))
    fire_iou  = _to_nan(np.array([m.fire_iou            for m in history], dtype=np.float64))
    ovl       = np.array([m.overlap_rate * 100           for m in history], dtype=np.float64)
    lat       = _to_nan(np.array([m.decision_latency_ms for m in history], dtype=np.float64))

    def _smooth(x, w=5):
        out  = np.full_like(x, np.nan)
        half = w // 2
        for i in range(len(x)):
            win   = x[max(0, i-half): i+half+1]
            valid = win[~np.isnan(win)]
            if len(valid): out[i] = valid.mean()
        return out

    fi_sm  = _smooth(front_iou)
    ri_sm  = _smooth(fire_iou)
    ovl_sm = _smooth(ovl)
    lat_sm = _smooth(lat)

    LINE_FIOU   = "#2ca02c"
    LINE_RIOU   = "#9467bd"
    LINE_OVL    = "#1f77b4"
    LINE_LAT    = "#ff7f0e"

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.0),
                              gridspec_kw={"wspace": 0.42})
    ax1, ax2, ax3, ax4 = axes

    def _style_ax(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(direction="in", length=3, width=0.8)
        ax.set_xlim(steps[0], steps[-1])

    def _iou_panel(ax, data, sm, color, title, label):
        _style_ax(ax)
        ax.fill_between(steps, np.nan_to_num(data, nan=0),
                        alpha=0.10, color=color, zorder=2)
        ax.plot(steps, data, color=color, lw=0.8, alpha=0.30, zorder=3)
        ax.plot(steps, sm,   color=color, lw=2.0,             zorder=4, label=label)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Simulation step")
        ax.set_ylabel("IoU")
        ax.set_title(title, loc="left", fontweight="bold", pad=5)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax.tick_params(which="minor", length=2, width=0.6)
        ax.legend(loc="lower right", frameon=True, framealpha=0.85,
                  edgecolor="#ccc", handlelength=1.5)

    _iou_panel(ax1, front_iou, fi_sm, LINE_FIOU, "(a) Frontier IoU", "Frontier IoU")
    _iou_panel(ax2, fire_iou,  ri_sm, LINE_RIOU, "(b) Fire IoU",     "Fire IoU")

    _style_ax(ax3)
    ax3.fill_between(steps, ovl, alpha=0.10, color=LINE_OVL, zorder=2)
    ax3.plot(steps, ovl,    color=LINE_OVL, lw=0.8, alpha=0.30, zorder=3)
    ax3.plot(steps, ovl_sm, color=LINE_OVL, lw=2.0,            zorder=4,
             label="Overlap rate")
    ax3.set_ylim(0, 105)
    ax3.set_xlabel("Simulation step")
    ax3.set_ylabel("Overlap rate (%)")
    ax3.set_title("(c) Sensor overlap", loc="left", fontweight="bold", pad=5)
    ax3.yaxis.set_major_locator(plt.MultipleLocator(20))
    ax3.yaxis.set_minor_locator(plt.MultipleLocator(10))
    ax3.tick_params(which="minor", length=2, width=0.6)
    ax3.legend(loc="upper right", frameon=True, framealpha=0.85,
               edgecolor="#ccc", handlelength=1.5)

    # Latency
    _style_ax(ax4)
    lat_valid = ~np.isnan(lat)
    ax4.scatter(steps[lat_valid], lat[lat_valid],
                s=6, color=LINE_LAT, alpha=0.40, zorder=3)
    ax4.plot(steps, lat_sm, color=LINE_LAT, lw=2.0, zorder=4,
             label="Mean latency")
    lat_finite = lat[lat_valid]
    y_top_lat  = (lat_finite.max() * 1.15) if len(lat_finite) else 10.0
    ax4.set_ylim(bottom=0, top=y_top_lat)
    ax4.set_xlabel("Simulation step")
    ax4.set_ylabel("Latency (ms)")
    ax4.set_title("(d) Decision latency", loc="left", fontweight="bold", pad=5)
    ax4.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax4.tick_params(which="minor", length=2, width=0.6)
    ax4.legend(loc="upper right", frameon=True, framealpha=0.85,
               edgecolor="#ccc", handlelength=1.5)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight",
                    format="pdf" if save_path.endswith(".pdf") else "png")
        print(f"Metrics plot saved → {save_path}")
    plt.show()
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)