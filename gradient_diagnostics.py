# gradient_diagnostics.py
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import json
from torch.cuda.amp import GradScaler


class GradientDiagnostics:
    """
    A diagnostics tool to monitor and analyze gradients in the MultiModalSpectralGPT model
    to help identify vanishing gradient problems.
    """

    def __init__(self, model, output_dir="gradient_diagnostics"):
        """
        Initialize the diagnostics tool.

        Args:
            model: The MultiModalSpectralGPT model to diagnose
            output_dir: Directory to save the diagnostic results
        """
        self.model = model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.log_file = os.path.join(output_dir, "gradient_diagnostics.txt")
        self.csv_file = os.path.join(output_dir, "gradient_stats.csv")

        # Storage for gradient statistics
        self.grad_stats = defaultdict(list)
        self.activation_stats = defaultdict(list)
        self.scaler_stats = []
        self.batch_counter = 0

        # Open log file
        self.log_fh = open(self.log_file, "w")
        self.log("Gradient Diagnostics initialized at {}".format(time.strftime("%Y-%m-%d %H:%M:%S")))

        # Layer categories for analysis
        self.layer_categories = {
            "encoder": ["patch_embed", "blocks", "cross_attn", "norm"],
            "decoder": ["decoder_embed", "decoder_blocks", "decoder_norm", "decoder_pred"],
            "projections": ["proj_head_global", "proj_head_spatial", "modality_proj"],
            "aux_encoders": ["aux_encoder"]
        }

        # Counters for hooks
        self.hook_handles = []

    def log(self, message):
        """Write a message to the log file and print it"""
        print(message)
        self.log_fh.write(message + "\n")
        self.log_fh.flush()

    def register_gradient_hooks(self):
        """Register gradient hooks on all model parameters"""
        self.log("Registering gradient hooks on model parameters")

        # Clear any existing hooks
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

        # Register new hooks
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                handle = param.register_hook(
                    lambda grad, name=name: self._gradient_hook(grad, name)
                )
                self.hook_handles.append(handle)

        self.log(f"Registered gradient hooks on {len(self.hook_handles)} parameters")
        return self

    def register_activation_hooks(self):
        """Register hooks to monitor activations in key layers"""
        self.log("Registering activation hooks on key layers")

        # Clear any existing hooks
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

        # Register forward hooks on key layers
        monitored_layers = []

        # Monitor decoder normalization layers
        for name, module in self.model.named_modules():
            if 'decoder' in name and isinstance(module, torch.nn.LayerNorm):
                handle = module.register_forward_hook(
                    lambda mod, inp, out, name=name: self._activation_hook(out, name)
                )
                self.hook_handles.append(handle)
                monitored_layers.append(name)

            # Also monitor the decoder prediction layer
            elif name == 'decoder_pred' and hasattr(module, 'forward'):
                handle = module.register_forward_hook(
                    lambda mod, inp, out, name=name: self._activation_hook(out, name)
                )
                self.hook_handles.append(handle)
                monitored_layers.append(name)

        self.log(f"Registered activation hooks on {len(monitored_layers)} layers: {monitored_layers}")
        return self

    def _gradient_hook(self, grad, name):
        """Hook function to capture gradient statistics"""
        if grad is None:
            return None

        # Calculate gradient statistics
        with torch.no_grad():
            grad_abs = grad.abs()
            stats = {
                "norm": torch.norm(grad).item(),
                "mean": grad.mean().item(),
                "std": grad.std().item(),
                "min": grad.min().item(),
                "max": grad.max().item(),
                "median": torch.median(grad_abs).item(),
                "zeros": (grad == 0).float().mean().item(),
                "name": name,
                "batch": self.batch_counter
            }

            # Categorize the parameter
            for category, keywords in self.layer_categories.items():
                if any(keyword in name for keyword in keywords):
                    stats["category"] = category
                    break
            else:
                stats["category"] = "other"

            self.grad_stats[name].append(stats)

        return grad

    def _activation_hook(self, output, name):
        """Hook function to capture activation statistics"""
        with torch.no_grad():
            if isinstance(output, torch.Tensor):
                stats = {
                    "mean": output.mean().item(),
                    "std": output.std().item(),
                    "min": output.min().item(),
                    "max": output.max().item(),
                    "zeros": (output == 0).float().mean().item(),
                    "name": name,
                    "batch": self.batch_counter
                }
                self.activation_stats[name].append(stats)
            elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                # Handle case where output is a tuple of tensors
                output = output[0]  # Take first tensor
                stats = {
                    "mean": output.mean().item(),
                    "std": output.std().item(),
                    "min": output.min().item(),
                    "max": output.max().item(),
                    "zeros": (output == 0).float().mean().item(),
                    "name": name,
                    "batch": self.batch_counter
                }
                self.activation_stats[name].append(stats)

        return None

    def track_gradscaler(self, scaler):
        """Track GradScaler statistics"""
        if isinstance(scaler, GradScaler):
            self.scaler_stats.append({
                "scale": scaler.get_scale(),
                "batch": self.batch_counter,
                "growth_tracker": scaler._get_growth_tracker() if hasattr(scaler, "_get_growth_tracker") else "N/A"
            })

    def analyze_batch(self, batch_idx, loss=None, scaler=None):
        """Analyze a batch after backward pass"""
        self.batch_counter = batch_idx

        # Track GradScaler if provided
        if scaler is not None:
            self.track_gradscaler(scaler)

        # Return self for chaining
        return self

    def _save_stats_to_csv(self):
        """Save all collected statistics to CSV files"""
        # Gradient stats
        all_grad_data = []
        for param_name, stats_list in self.grad_stats.items():
            all_grad_data.extend(stats_list)

        if all_grad_data:
            grad_df = pd.DataFrame(all_grad_data)
            grad_df.to_csv(os.path.join(self.output_dir, "gradient_stats.csv"), index=False)

        # Activation stats
        all_activation_data = []
        for layer_name, stats_list in self.activation_stats.items():
            all_activation_data.extend(stats_list)

        if all_activation_data:
            activation_df = pd.DataFrame(all_activation_data)
            activation_df.to_csv(os.path.join(self.output_dir, "activation_stats.csv"), index=False)

        # GradScaler stats
        if self.scaler_stats:
            scaler_df = pd.DataFrame(self.scaler_stats)
            scaler_df.to_csv(os.path.join(self.output_dir, "gradscaler_stats.csv"), index=False)

    def generate_report(self):
        """Generate a comprehensive diagnostic report"""
        self.log("\n" + "=" * 50)
        self.log("Generating Gradient Diagnostics Report")
        self.log("=" * 50)

        # 1. Save all collected statistics to CSV
        self._save_stats_to_csv()

        # 2. Generate plots
        self._generate_plots()

        # 3. Check for gradient vanishing signs
        vanishing_status = self._check_for_vanishing_gradients()

        # 4. Summarize findings in the report
        self._write_summary_report(vanishing_status)

        self.log("Report generation complete. Files saved to " + self.output_dir)
        self.log_fh.close()

        return vanishing_status

    def _generate_plots(self):
        """Generate diagnostic plots"""
        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # 1. Gradient norms by layer type
        self._plot_gradient_norms_by_category(plots_dir)

        # 2. Gradient distributions
        self._plot_gradient_distributions(plots_dir)

        # 3. Activation statistics
        self._plot_activation_stats(plots_dir)

        # 4. GradScaler tracking
        self._plot_gradscaler_stats(plots_dir)

    def _plot_gradient_norms_by_category(self, plots_dir):
        """Plot gradient norms by layer category"""
        # Collect data by category
        all_grad_data = []
        for param_name, stats_list in self.grad_stats.items():
            all_grad_data.extend(stats_list)

        if not all_grad_data:
            return

        df = pd.DataFrame(all_grad_data)

        # Skip if we don't have enough data
        if len(df) == 0 or "category" not in df.columns:
            return

        # Plot
        plt.figure(figsize=(12, 8))

        # Group by category and batch
        category_data = df.groupby(['category', 'batch'])['norm'].mean().reset_index()

        # Plot each category
        for category in df['category'].unique():
            category_df = category_data[category_data['category'] == category]
            plt.plot(category_df['batch'], category_df['norm'], 'o-', label=category)

        plt.title('Gradient Norms by Layer Category')
        plt.xlabel('Batch')
        plt.ylabel('Mean Gradient Norm')
        plt.legend()
        plt.yscale('log')  # Log scale for better visualization
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "gradient_norms_by_category.png"))
        plt.close()

    def _plot_gradient_distributions(self, plots_dir):
        """Plot distributions of gradients for different layer types"""
        # Collect data by category
        all_grad_data = []
        for param_name, stats_list in self.grad_stats.items():
            all_grad_data.extend(stats_list)

        if not all_grad_data:
            return

        df = pd.DataFrame(all_grad_data)

        # Skip if we don't have enough data
        if len(df) == 0 or "category" not in df.columns:
            return

        # Plot histogram for last batch
        if "batch" in df.columns:
            last_batch = df['batch'].max()
            last_batch_df = df[df['batch'] == last_batch]

            plt.figure(figsize=(12, 8))

            for i, category in enumerate(last_batch_df['category'].unique()):
                category_df = last_batch_df[last_batch_df['category'] == category]
                plt.subplot(2, 2, i + 1)
                plt.hist(category_df['norm'], bins=20, alpha=0.7, label=category)
                plt.title(f'Gradient Norm Distribution - {category}')
                plt.xlabel('Gradient Norm')
                plt.ylabel('Count')
                plt.grid(True, alpha=0.3)
                plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "gradient_distribution_by_category.png"))
            plt.close()

    def _plot_activation_stats(self, plots_dir):
        """Plot activation statistics"""
        # Collect all activation data
        all_activation_data = []
        for layer_name, stats_list in self.activation_stats.items():
            all_activation_data.extend(stats_list)

        if not all_activation_data:
            return

        df = pd.DataFrame(all_activation_data)

        # Skip if we don't have enough data
        if len(df) == 0:
            return

        # Plot activation means
        plt.figure(figsize=(12, 8))

        grouped = df.groupby(['name', 'batch']).agg({
            'mean': 'mean',
            'std': 'mean',
            'min': 'min',
            'max': 'max'
        }).reset_index()

        for name in grouped['name'].unique():
            layer_data = grouped[grouped['name'] == name]
            plt.plot(layer_data['batch'], layer_data['mean'], 'o-', label=f"{name} (mean)")

        plt.title('Activations Mean by Layer')
        plt.xlabel('Batch')
        plt.ylabel('Activation Mean')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "activation_means.png"))
        plt.close()

        # Plot activation std
        plt.figure(figsize=(12, 8))

        for name in grouped['name'].unique():
            layer_data = grouped[grouped['name'] == name]
            plt.plot(layer_data['batch'], layer_data['std'], 'o-', label=f"{name} (std)")

        plt.title('Activations Standard Deviation by Layer')
        plt.xlabel('Batch')
        plt.ylabel('Activation Std')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "activation_std.png"))
        plt.close()

    def _plot_gradscaler_stats(self, plots_dir):
        """Plot GradScaler statistics"""
        if not self.scaler_stats:
            return

        df = pd.DataFrame(self.scaler_stats)

        plt.figure(figsize=(10, 6))
        plt.plot(df['batch'], df['scale'], 'o-', label='Scale')
        plt.title('GradScaler Scale Factor')
        plt.xlabel('Batch')
        plt.ylabel('Scale')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "gradscaler_scale.png"))
        plt.close()

    def _check_for_vanishing_gradients(self):
        """Check for signs of vanishing gradients"""
        results = {
            "has_vanishing_gradients": False,
            "warnings": [],
            "encoder_decoder_ratio": None,
            "gradscaler_issues": False
        }

        # Collect all gradient data
        all_grad_data = []
        for param_name, stats_list in self.grad_stats.items():
            all_grad_data.extend(stats_list)

        if not all_grad_data:
            results["warnings"].append("No gradient data collected to analyze")
            return results

        df = pd.DataFrame(all_grad_data)

        # Skip if we don't have enough data
        if len(df) == 0 or "category" not in df.columns:
            results["warnings"].append("Insufficient gradient data for analysis")
            return results

        # Analysis by category
        if "batch" in df.columns:
            last_batch = df['batch'].max()
            last_batch_df = df[df['batch'] == last_batch]

            # Calculate average gradient norm by category
            category_norms = {}
            for category in last_batch_df['category'].unique():
                category_df = last_batch_df[last_batch_df['category'] == category]
                category_norms[category] = category_df['norm'].mean()

            # Check for vanishing gradients in decoder vs encoder
            if 'encoder' in category_norms and 'decoder' in category_norms:
                encoder_norm = category_norms['encoder']
                decoder_norm = category_norms['decoder']
                ratio = decoder_norm / (encoder_norm + 1e-10)
                results["encoder_decoder_ratio"] = ratio

                if ratio < 0.1:
                    results["has_vanishing_gradients"] = True
                    results["warnings"].append(
                        f"Decoder gradients ({decoder_norm:.8f}) are much smaller than encoder gradients ({encoder_norm:.8f}), "
                        f"ratio: {ratio:.8f}. This strongly suggests vanishing gradients."
                    )

            # Check for very small gradients in decoder
            if 'decoder' in category_norms and category_norms['decoder'] < 1e-6:
                results["has_vanishing_gradients"] = True
                results["warnings"].append(
                    f"Decoder gradients are extremely small: {category_norms['decoder']:.8f}. "
                    f"This indicates vanishing gradients."
                )

            # Check for vanishing gradients by depth
            decoder_layers = {}
            for name, stats_list in self.grad_stats.items():
                if 'decoder_blocks' in name:
                    # Try to extract layer number
                    try:
                        layer_num = int(''.join(filter(str.isdigit, name.split('decoder_blocks.')[1].split('.')[0])))
                        if stats_list and len(stats_list) > 0:
                            last_stat = stats_list[-1]
                            decoder_layers[layer_num] = last_stat['norm']
                    except:
                        pass

            # Check for decreasing gradient norms with depth
            if len(decoder_layers) > 2:
                layer_nums = sorted(decoder_layers.keys())
                first_layer_norm = decoder_layers[layer_nums[0]]
                last_layer_norm = decoder_layers[layer_nums[-1]]

                if last_layer_norm < first_layer_norm * 0.1:
                    results["has_vanishing_gradients"] = True
                    results["warnings"].append(
                        f"Gradient norm decreases significantly with decoder depth: "
                        f"first layer: {first_layer_norm:.8f}, last layer: {last_layer_norm:.8f}. "
                        f"Ratio: {last_layer_norm / first_layer_norm:.8f}. This indicates vanishing gradients."
                    )

        # Analyze GradScaler if available
        if self.scaler_stats:
            scaler_df = pd.DataFrame(self.scaler_stats)

            # Check if scale is decreasing
            if len(scaler_df) >= 2:
                first_scale = scaler_df['scale'].iloc[0]
                last_scale = scaler_df['scale'].iloc[-1]

                if last_scale < first_scale * 0.1:
                    results["gradscaler_issues"] = True
                    results["warnings"].append(
                        f"GradScaler scale has decreased significantly from {first_scale} to {last_scale}. "
                        f"This indicates numerical instability in gradients, possibly due to vanishing or exploding gradients."
                    )

        return results

    def _write_summary_report(self, vanishing_status):
        """Write a summary report based on the analysis"""
        summary_file = os.path.join(self.output_dir, "summary_report.txt")

        with open(summary_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("GRADIENT DIAGNOSTICS SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write("VANISHING GRADIENTS ANALYSIS:\n")
            f.write("-" * 50 + "\n")

            if vanishing_status["has_vanishing_gradients"]:
                f.write("VERDICT: VANISHING GRADIENTS DETECTED\n\n")
            else:
                f.write("VERDICT: NO CLEAR SIGN OF VANISHING GRADIENTS\n\n")

            f.write("Warnings and Issues:\n")
            if vanishing_status["warnings"]:
                for i, warning in enumerate(vanishing_status["warnings"], 1):
                    f.write(f"{i}. {warning}\n")
            else:
                f.write("No specific issues detected.\n")

            f.write("\n")

            if vanishing_status["encoder_decoder_ratio"] is not None:
                f.write(f"Encoder-Decoder Gradient Ratio: {vanishing_status['encoder_decoder_ratio']:.8f}\n")
                f.write(f"(Healthy models typically have ratios between 0.1 and 10)\n\n")

            # GradScaler analysis
            f.write("GRADIENT SCALER ANALYSIS:\n")
            f.write("-" * 50 + "\n")

            if vanishing_status["gradscaler_issues"]:
                f.write("GradScaler issues detected - possible numerical instability.\n")
            else:
                f.write("No GradScaler issues detected.\n")

            if self.scaler_stats:
                scaler_df = pd.DataFrame(self.scaler_stats)
                f.write(f"Initial scale: {scaler_df['scale'].iloc[0]}\n")
                f.write(f"Final scale: {scaler_df['scale'].iloc[-1]}\n")

            f.write("\n")

            # General recommendations
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 50 + "\n")

            if vanishing_status["has_vanishing_gradients"]:
                f.write("1. Consider using residual connections in your decoder\n")
                f.write("2. Try Layer Normalization before self-attention in transformer blocks\n")
                f.write("3. Adjust learning rates - use separate optimizer groups with higher LR for decoder\n")
                f.write("4. Reduce the depth of your decoder network\n")
                f.write("5. Use gradient clipping to stabilize training\n")
                f.write("6. Consider reducing the diversity loss weights initially in training\n")
            else:
                f.write("Your model seems to have healthy gradient flow. If you're experiencing patch collapse,\n")
                f.write("the issue may be related to:\n")
                f.write("1. Loss function design or weighting\n")
                f.write("2. Initialization issues\n")
                f.write("3. Batch normalization or activation function problems\n")
                f.write("4. Dataset imbalance or preprocessing issues\n")

            # Save raw data reference
            f.write("\n")
            f.write("RAW DATA:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Detailed gradient statistics: {os.path.join(self.output_dir, 'gradient_stats.csv')}\n")
            f.write(f"Activation statistics: {os.path.join(self.output_dir, 'activation_stats.csv')}\n")
            if self.scaler_stats:
                f.write(f"GradScaler statistics: {os.path.join(self.output_dir, 'gradscaler_stats.csv')}\n")
            f.write(f"Diagnostic plots: {os.path.join(self.output_dir, 'plots')}\n")

        # Also save a JSON version of the analysis
        with open(os.path.join(self.output_dir, "vanishing_gradient_analysis.json"), "w") as f:
            json.dump(vanishing_status, f, indent=2)

        return summary_file