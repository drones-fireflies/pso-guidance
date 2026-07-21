import numpy as np
import matplotlib.pyplot as plt


# Matplotlib defaults tuned for compact, crisp figures
plt.rcParams.update({
    "font.size": 3,
    "axes.labelsize": 4,
    "axes.titlesize": 5,
    "xtick.labelsize": 4,
    "ytick.labelsize": 4,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.edgecolor": "0.3",
    "axes.linewidth": 0.5,
})


class Display:
    """

    Parameters
    ----------
    height, width : int
        Grid dimensions.
    wind_direction : float
        Wind direction in radians (0 = east, pi/2 = north).
    elevation : np.ndarray
        Elevation map.

    """

    def __init__(self, height: int, width: int, wind_direction: float, elevation: np.ndarray) -> None:
        self.height = height
        self.width = width
        self.wind_direction = float(wind_direction)

        # Wind field components
        self.wind_u = np.cos(self.wind_direction)
        self.wind_v = np.sin(self.wind_direction)

        # Storage
        self.elevation = elevation

    def _initialize_visualization(self):
        # Create figure and axes
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes = axes.ravel()
        fig.subplots_adjust(wspace=0.3, hspace=0.3, top=0.85)

        fig.suptitle(f"Wind Direction: {self.wind_direction:.2f} rad", fontsize=6, y=0.95)

        subplot_titles = ["Fire", "Fuel"]
        images_dict = {}

        def add_subplot_image(ax, title, cmap, vmin=0, vmax=1):
            """Function to add an image to a subplot with a colorbar."""
            img = ax.imshow(np.zeros((self.height, self.width)), cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title)
            fig.colorbar(img, ax=ax, fraction=0.036, pad=0.04)
            return img

        # Initialize each subplot
        images_dict["fire"] = add_subplot_image(axes[0], subplot_titles[0], cmap="viridis")
        images_dict["fuel"] = add_subplot_image(axes[1], subplot_titles[1], cmap="YlGn", vmin=0)

        return fig, images_dict


    def update(self, images, fire, fuel):
        """Update the displayed images during simulation."""
        images["fire"].set_data(fire)
        images["fuel"].set_data(fuel)
        plt.pause(0.01)


    def show(self, fire_history) -> None:
        """Show final fire spread map."""
        
        n_steps = len(fire_history)
        height, width = self.height, self.width
        elev = self.elevation

        # Compute fire arrival time map
        fire_stack = np.array(fire_history) > 0
        arrival_time = np.zeros((height, width))
        for t in range(n_steps):
            new_fire = (fire_stack[t]) & (arrival_time == 0)
            arrival_time[new_fire] = t + 1

        burned = arrival_time > 0

        # ------------------------------------------------------------------
        # Plot setup
        # ------------------------------------------------------------------
        fig, ax = plt.subplots()
        fig.subplots_adjust(wspace=0.3, hspace=0.3, top=0.85)
        ax.set_title("Fire Spread Map", fontsize=5)
        ax.set_xlabel("South")
        ax.set_ylabel("Ouest")

        # Terrain background
        ax.imshow(elev, cmap="terrain", origin="upper", vmin=0, vmax=1)

        # More contour levels for finer gradients
        vmin, vmax = np.min(arrival_time[burned]), np.max(arrival_time[burned])
        levels = np.linspace(vmin, vmax, 10)

        # Filled contours
        filled = ax.contourf(arrival_time, levels=levels, cmap="Oranges", alpha=0.7)

        # Wind quiver field
        yy, xx = np.mgrid[0:height, 0:width]
        step = max(1, int(min(height, width) // 40))
        ax.quiver(xx[::step, ::step], yy[::step, ::step],
                  self.wind_u * np.ones_like(xx[::step, ::step]),
                  self.wind_v * np.ones_like(yy[::step, ::step]),
                  scale=70, width=0.002, color="dimgrey", alpha=0.8)

        # Ignition point(s)
        first_t = np.min(arrival_time[burned])
        iy, ix = np.where(arrival_time == first_t)
        if len(ix) > 0:
            ax.scatter(ix, iy, s=5, c="red", marker="o", edgecolors="black",
                       linewidths=0.2, label="Ignition point")

        # Colorbar with better ticks and label
        # cbar = plt.colorbar(filled, ax=ax, fraction=0.03, pad=0.04)
        # cbar.set_label("Fire arrival time (timesteps)", fontsize=3)
        # cbar.ax.tick_params(labelsize=3)

        plt.tight_layout()
        plt.show()