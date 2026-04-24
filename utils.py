import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_diagnostics(metrics, title, filename):
    """Plot training diagnostics.

    Four subplots: policy loss, value loss

    Args:
        metrics: dict with keys 'policy_loss', 'value_loss', 'dur_loss' (each a list of floats)
        title: Plot title prefix
        filename: Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    panels = [
        ('policy_loss', 'Policy Loss'),
        ('value_loss', 'Value Loss'),
        ('dur_loss', 'Duration Loss'),
    ]

    for ax, (key, label) in zip(axes.flat, panels):
        if key in metrics and metrics[key]:
            ax.plot(metrics[key], linewidth=1.5, color="steelblue")
            ax.set_title(f"{title} \u2014 {label}", fontsize=11)
            ax.set_xlabel("Iteration", fontsize=10)
            ax.set_ylabel(label, fontsize=10)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()