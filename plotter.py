from utils import plot_diagnostics, plot_general
import argparse
import numpy as np

def get_all_params(filename):
    data = np.load(filename, allow_pickle=True)
    policy_loss = data['policy_loss_train']
    value_loss = data['value_loss_train']
    dur_loss = data['duration_loss_train']
    entropy = data['entropy']
    eval_rewards = data['eval_rewards']
    metrics = {'policy_loss': policy_loss, 'value_loss': value_loss, 'dur_loss': dur_loss,
               'entropy': entropy, 'eval_rewards': eval_rewards}
    return metrics

def plot(filename, diagnostic=False, eval=False):
    if diagnostic:
        plot_diagnostics(filename, metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("PPO Pointer Network")
    parser.add_argument("save_file", help="Parameter name", type=str)
    parser.add_argument("image_file", help="Parameter name", type=str)
    args = parser.parse_args()
    save_file = args.save_file + '.pth'
    image_file = args.image_file + 'png'
    metrics = get_all_params(save_file)
    plot(image_file, diagnostic=True)