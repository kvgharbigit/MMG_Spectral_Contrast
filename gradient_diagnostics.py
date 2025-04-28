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
    An improved diagnostics tool to monitor and analyze gradients in the MultiModalSpectralGPT model
    to help identify vanishing gradient problems across the entire model.
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

        # Store named parameters for easier access later
        self.named_parameters = list(model.named_parameters())
        self.param_dict = {name: param for name, param in self.named_parameters}

        # Open log file
        self.log_fh = open(self.log_file, "w")
        self.log(f"Gradient Diagnostics initialized at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Model has {len(self.named_parameters)} named parameters")

        # Layer categories for analysis
        self.layer_categories = {
            "encoder": ["patch_embed", "pos_embed", "blocks", "cross_attn", "norm"],
            "decoder": ["decoder_embed", "decoder_blocks", "decoder_norm", "decoder_pred", "decoder_pos_embed"],
            "projections": ["proj_head_global", "proj_head_spatial", "modality_proj"],
            "aux_encoders": ["aux_encoder"],
            "contrastive": ["temperature", "mask_token"]
        }

        # Count parameters by category
        self.count_parameters_by_category()

        # Counters for hooks
        self.hook_handles = []

    def count_parameters_by_category(self):
        """Count parameters by category and log the results"""
        category_counts = defaultdict(int)

        for name, param in self.named_parameters:
            if param.requires_grad:
                category = "other"
                for cat_name, keywords in self.layer_categories.items():
                    if any(keyword in name for keyword in keywords):
                        category = cat_name
                        break
                category_counts[category] += 1

        self.log("Parameter counts by category:")
        for category, count in category_counts.items():
            self.log(f"  - {category}: {count} parameters")

        return category_counts

    def log(self, message):
        """Write a message to the log file and print it"""
        print(message)
        self.log_fh.write(message + "\n")
        self.log_fh.flush()

    def register_gradient_hooks(self):
        """Register gradient hooks on all model parameters - improved version"""
        self.log("Registering gradient hooks on model parameters")

        # Clear any existing hooks
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

        # Register new hooks - count parameters by gradient requirement
        requires_grad_count = 0
        no_grad_count = 0

        # Register hooks for each parameter - use a proper hook function
        for name, param in self.named_parameters:
            if param.requires_grad:
                requires_grad_count += 1

                # Define hook function with fixed name parameter
                def hook_fn(grad, param_name=name):
                    return self._gradient_hook(grad, param_name)

                handle = param.register_hook(hook_fn)
                self.hook_handles.append(handle)
            else:
                no_grad_count += 1

        self.log(f"Registered gradient hooks on {requires_grad_count} parameters")
        self.log(f"Skipped {no_grad_count} parameters that don't require gradients")
        return self

    def register_activation_hooks(self):
        """Register hooks to monitor activations in both encoder and decoder key layers"""
        self.log("Registering activation hooks on key layers across the model")

        # Clear any existing hooks
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

        # Register forward hooks on key layers
        monitored_layers = []

        # Monitor normalization layers and important components in both encoder and decoder
        for name, module in self.model.named_modules():
            # Check all normalization layers
            if isinstance(module, torch.nn.LayerNorm):
                # Define hook function with fixed name parameter
                def activation_hook_fn(mod, inp, out, layer_name=name):
                    return self._activation_hook(out, layer_name)

                handle = module.register_forward_hook(activation_hook_fn)
                self.hook_handles.append(handle)
                monitored_layers.append(name)

            # Check prediction, projection and other key layers
            elif any(key in name for key in ['pred', 'proj_head', 'patch_embed']) and hasattr(module, 'forward'):
                # Define hook function with fixed name parameter
                def projection_hook_fn(mod, inp, out, layer_name=name):
                    return self._activation_hook(out, layer_name)

                handle = module.register_forward_hook(projection_hook_fn)
                self.hook_handles.append(handle)
                monitored_layers.append(name)

        self.log(f"Registered activation hooks on {len(monitored_layers)} layers")
        return self

    def _gradient_hook(self, grad, name):
        """Hook function to capture gradient statistics - improved version"""
        if grad is None:
            self.log(f"Warning: Gradient is None for parameter {name}")
            return None

        # Calculate gradient statistics
        with torch.no_grad():
            grad_abs = grad.abs()

            # Handle different tensor shapes
            if grad.dim() > 1:
                flat_grad = grad.view(-1)
                median_val = torch.median(flat_grad)
            else:
                median_val = torch.median(grad)

            stats = {
                "norm": torch.norm(grad).item(),
                "mean": grad.mean().item(),
                "std": grad.std().item(),
                "min": grad.min().item(),
                "max": grad.max().item(),
                "median": median_val.item(),
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

            # Store the stats
            self.grad_stats[name].append(stats)

        # The hook must return the gradient unchanged
        return grad

    def _activation_hook(self, output, name):
        """Hook function to capture activation statistics - improved version"""
        with torch.no_grad():
            if isinstance(output, torch.Tensor):
                # Calculate statistics
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

        # No need to return anything for activation hooks
        return None

    def track_gradscaler(self, scaler):
        """Track GradScaler statistics"""
        if isinstance(scaler, GradScaler):
            # Store scaler stats
            self.scaler_stats.append({
                "scale": scaler.get_scale(),
                "batch": self.batch_counter,
                "growth_tracker": getattr(scaler, "_growth_tracker", 0)
            })
            self.log(f"Tracked GradScaler with scale: {scaler.get_scale()}")

    def analyze_batch(self, batch_idx, loss=None, scaler=None):
        """Analyze a batch after backward pass"""
        self.batch_counter = batch_idx
        self.log(f"Analyzing batch {batch_idx}")

        # Track GradScaler if provided
        if scaler is not None:
            self.track_gradscaler(scaler)

        # Check gradient statistics directly
        self.check_gradients_directly()

        # Check if we have gradients for each layer type
        self._check_gradient_coverage()

        # Return self for chaining
        return self

    def check_gradients_directly(self):
        """Check gradients directly from model parameters"""
        # Count how many parameters have non-None gradients
        grad_count = 0
        total_params = 0
        grad_categories = defaultdict(int)

        # Also collect some sample gradient info
        sample_grads = []

        for name, param in self.named_parameters:
            if param.requires_grad:
                total_params += 1

                # Check if gradient exists and is not None
                if param.grad is not None:
                    grad_count += 1

                    # Categorize
                    category = "other"
                    for cat_name, keywords in self.layer_categories.items():
                        if any(keyword in name for keyword in keywords):
                            category = cat_name
                            break

                    grad_categories[category] += 1

                    # Add to sample if it's an interesting layer
                    if 'decoder_pred' in name or 'blocks.0' in name or 'decoder_blocks.0' in name:
                        grad_mean = param.grad.abs().mean().item()
                        sample_grads.append(f"{name}: {grad_mean:.8f}")

        # Log the results
        self.log(f"Direct gradient check: {grad_count}/{total_params} parameters have gradients")

        # Log categories
        for category, count in grad_categories.items():
            self.log(f"  - {category}: {count} parameters have gradients")

        # Log sample gradients
        if sample_grads:
            self.log(f"Sample gradients: {', '.join(sample_grads[:10])}")
        else:
            self.log(f"No sample gradients found")

        return grad_count, total_params

    def _check_gradient_coverage(self):
        """Check if we have gradients for all major components of the model"""
        # Get all the parameters that should have gradients
        param_count = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_count += 1

        # Get the parameters we collected gradients for
        grad_count = len(self.grad_stats)

        # If we have significantly fewer gradients than parameters, log a warning
        if grad_count < param_count * 0.5:  # Less than half
            self.log(f"Warning: Only collected gradients for {grad_count}/{param_count} parameters")

            # Check which major components we're missing
            missing_categories = []
            for category in self.layer_categories:
                category_found = False
                for param_name in self.grad_stats:
                    for keyword in self.layer_categories[category]:
                        if keyword in param_name:
                            category_found = True
                            break
                    if category_found:
                        break

                if not category_found:
                    missing_categories.append(category)

            if missing_categories:
                self.log(f"Missing gradients for entire categories: {missing_categories}")
        else:
            self.log(f"Good gradient coverage: {grad_count}/{param_count} parameters")

    def _save_stats_to_csv(self):
        """Save all collected statistics to CSV files"""
        # Gradient stats
        all_grad_data = []
        for param_name, stats_list in self.grad_stats.items():
            all_grad_data.extend(stats_list)

        if all_grad_data:
            grad_df = pd.DataFrame(all_grad_data)
            grad_df.to_csv(os.path.join(self.output_dir, "gradient_stats.csv"), index=False)
            # Print categories found
            if 'category' in grad_df.columns:
                categories = grad_df['category'].unique()
                self.log(f"Found gradient data for categories: {list(categories)}")
                # Count parameters per category
                category_counts = grad_df.groupby('category')['name'].nunique()
                for category, count in category_counts.items():
                    self.log(f"  - {category}: {count} parameters")

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
        summary_path = self._write_summary_report(vanishing_status)

        self.log(f"Report generation complete. Files saved to {self.output_dir}")
        self.log_fh.close()

        return vanishing_status, summary_path

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

        # 5. Encoder vs Decoder gradient comparisons
        self._plot_encoder_decoder_comparison(plots_dir)

    def _plot_gradient_norms_by_category(self, plots_dir):
        """Plot gradient norms by layer category"""
        # Collect data by category
        all_grad_data = []
        for param_name, stats_list in self.grad_stats.items():
            all_grad_data.extend(stats_list)

        if not all_grad_data:
            self.log("No gradient data available for norm by category plot")
            return

        df = pd.DataFrame(all_grad_data)

        # Skip if we don't have enough data
        if len(df) == 0 or "category" not in df.columns:
            self.log("Insufficient gradient data for norm by category plot")
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

    def _plot_encoder_decoder_comparison(self, plots_dir):
        """Plot direct comparison of encoder vs decoder gradients"""
        all_grad_data = []
        for param_name, stats_list in self.grad_stats.items():
            all_grad_data.extend(stats_list)

        if not all_grad_data:
            self.log("No gradient data available for encoder-decoder comparison")
            return

        df = pd.DataFrame(all_grad_data)

        # Only continue if we have both encoder and decoder data
        if 'category' not in df.columns or not {'encoder', 'decoder'}.issubset(set(df['category'].unique())):
            self.log("Missing encoder or decoder data for comparison plot")
            return

        # Group by category and batch, calculate various statistics
        encoder_data = df[df['category'] == 'encoder'].groupby('batch')['norm'].agg(
            ['mean', 'min', 'max']).reset_index()
        decoder_data = df[df['category'] == 'decoder'].groupby('batch')['norm'].agg(
            ['mean', 'min', 'max']).reset_index()

        # Make sure we have data for both
        if len(encoder_data) == 0 or len(decoder_data) == 0:
            self.log("Insufficient data for encoder-decoder comparison")
            return

        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 14), sharex=True)

        # Plot mean gradients
        axes[0].plot(encoder_data['batch'], encoder_data['mean'], 'o-', label='Encoder', color='blue')
        axes[0].plot(decoder_data['batch'], decoder_data['mean'], 'o-', label='Decoder', color='red')
        axes[0].set_title('Mean Gradient Norm: Encoder vs Decoder')
        axes[0].set_ylabel('Mean Gradient Norm')
        axes[0].set_yscale('log')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot ratio of decoder/encoder for each batch
        merged_data = pd.merge(encoder_data, decoder_data, on='batch', suffixes=('_encoder', '_decoder'))
        merged_data['ratio'] = merged_data['mean_decoder'] / merged_data['mean_encoder']

        axes[1].plot(merged_data['batch'], merged_data['ratio'], 'o-', color='purple')
        axes[1].axhline(y=1.0, color='green', linestyle='--')
        axes[1].set_title('Decoder/Encoder Gradient Ratio (1.0 = Equal)')
        axes[1].set_ylabel('Ratio')
        axes[1].set_xlabel('Batch')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "encoder_decoder_comparison.png"))
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

            plt.figure(figsize=(15, 10))

            categories = last_batch_df['category'].unique()
            num_categories = len(categories)

            # Calculate grid dimensions
            grid_size = int(np.ceil(np.sqrt(num_categories)))
            rows = grid_size
            cols = grid_size

            for i, category in enumerate(categories):
                if i < rows * cols:  # Make sure we don't exceed subplot grid
                    plt.subplot(rows, cols, i + 1)
                    category_df = last_batch_df[last_batch_df['category'] == category]
                    plt.hist(category_df['norm'], bins=20, alpha=0.7, label=category)
                    plt.title(f'Gradient Norm - {category}')
                    plt.xlabel('Gradient Norm')
                    plt.ylabel('Count')
                    plt.grid(True, alpha=0.3)
                    plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "gradient_distribution_by_category.png"))
            plt.close()

            # Also create a layer depth analysis for decoder and encoder
            self._plot_gradient_depth_analysis(df, last_batch, plots_dir)

    def _plot_gradient_depth_analysis(self, df, last_batch, plots_dir):
        """Analyze and plot gradients by depth in encoder and decoder"""
        last_batch_df = df[df['batch'] == last_batch].copy()

        # Extract layer indices where possible
        def extract_index(name, pattern):
            import re
            match = re.search(f"{pattern}\\.([0-9]+)", name)
            if match:
                return int(match.group(1))
            return None

        # Process encoder blocks
        encoder_data = []
        for _, row in last_batch_df.iterrows():
            if 'blocks' in row['name']:
                layer_idx = extract_index(row['name'], 'blocks')
                if layer_idx is not None:
                    encoder_data.append({'layer': layer_idx, 'norm': row['norm']})

        # Process decoder blocks
        decoder_data = []
        for _, row in last_batch_df.iterrows():
            if 'decoder_blocks' in row['name']:
                layer_idx = extract_index(row['name'], 'decoder_blocks')
                if layer_idx is not None:
                    decoder_data.append({'layer': layer_idx, 'norm': row['norm']})

        # Only create plots if we have data
        if encoder_data or decoder_data:
            plt.figure(figsize=(15, 7))

            # Plot encoder gradient by depth if available
            if encoder_data:
                encoder_df = pd.DataFrame(encoder_data)
                encoder_summary = encoder_df.groupby('layer')['norm'].mean().reset_index()
                plt.subplot(1, 2, 1)
                plt.plot(encoder_summary['layer'], encoder_summary['norm'], 'o-', label='Encoder Blocks')
                plt.title('Encoder Gradient Norm by Depth')
                plt.xlabel('Block Index')
                plt.ylabel('Mean Gradient Norm')
                plt.grid(True, alpha=0.3)

            # Plot decoder gradient by depth if available
            if decoder_data:
                decoder_df = pd.DataFrame(decoder_data)
                decoder_summary = decoder_df.groupby('layer')['norm'].mean().reset_index()
                plt.subplot(1, 2, 2)
                plt.plot(decoder_summary['layer'], decoder_summary['norm'], 'o-', label='Decoder Blocks')
                plt.title('Decoder Gradient Norm by Depth')
                plt.xlabel('Block Index')
                plt.ylabel('Mean Gradient Norm')
                plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "gradient_norm_by_depth.png"))
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

        # Categorize layers into encoder/decoder for comparison
        layer_categories = {}
        for name in grouped['name'].unique():
            if 'decoder' in name:
                layer_categories[name] = 'decoder'
            elif any(keyword in name for keyword in ['norm', 'block', 'embed']):
                layer_categories[name] = 'encoder'
            else:
                layer_categories[name] = 'other'

        # Add category to the grouped data
        grouped['category'] = grouped['name'].map(layer_categories)

        # Plot activations by category (encoder vs decoder)
        plt.figure(figsize=(12, 8))

        for category in ['encoder', 'decoder', 'other']:
            category_data = grouped[grouped['category'] == category]
            if len(category_data) > 0:
                category_means = category_data.groupby('batch')['mean'].mean()
                plt.plot(category_means.index, category_means.values, 'o-', label=f"{category}")

        plt.title('Average Activations by Component Type')
        plt.xlabel('Batch')
        plt.ylabel('Mean Activation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "activation_by_category.png"))
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
            "gradscaler_issues": False,
            "gradient_magnitude_issues": False,
            "layer_depth_issues": False
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

        # Check gradient coverage
        categories_found = set(df['category'].unique())
        expected_categories = set(self.layer_categories.keys())
        missing_categories = expected_categories - categories_found

        if missing_categories:
            results["warnings"].append(
                f"Missing gradient data for categories: {missing_categories}. "
                f"Found data only for: {categories_found}"
            )

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

                # If either is extremely small, flag it directly
                if encoder_norm < 1e-6:
                    results["has_vanishing_gradients"] = True
                    results["gradient_magnitude_issues"] = True
                    results["warnings"].append(
                        f"Encoder gradients are extremely small: {encoder_norm:.8f}. "
                        f"This indicates vanishing gradients."
                    )

                if decoder_norm < 1e-6:
                    results["has_vanishing_gradients"] = True
                    results["gradient_magnitude_issues"] = True
                    results["warnings"].append(
                        f"Decoder gradients are extremely small: {decoder_norm:.8f}. "
                        f"This indicates vanishing gradients."
                    )

                # Calculate and check ratio
                ratio = decoder_norm / (encoder_norm + 1e-10)
                results["encoder_decoder_ratio"] = ratio

                if ratio < 0.1:
                    results["has_vanishing_gradients"] = True
                    results["warnings"].append(
                        f"Decoder gradients ({decoder_norm:.8f}) are much smaller than encoder gradients ({encoder_norm:.8f}), "
                        f"ratio: {ratio:.8f}. This strongly suggests vanishing gradients in the decoder."
                    )
                elif ratio > 10:
                    results["warnings"].append(
                        f"Decoder gradients ({decoder_norm:.8f}) are much larger than encoder gradients ({encoder_norm:.8f}), "
                        f"ratio: {ratio:.8f}. This suggests potential issues with gradient flow in the encoder."
                    )

            # Check for vanishing gradients by depth
            encoder_layers = {}
            decoder_layers = {}

            # Extract layer indices for encoder blocks
            for name, stats_list in self.grad_stats.items():
                if 'blocks' in name and not 'decoder' in name:
                    try:
                        layer_num = int(''.join(filter(str.isdigit, name.split('blocks.')[1].split('.')[0])))
                        if stats_list and len(stats_list) > 0:
                            last_stat = stats_list[-1]
                            encoder_layers[layer_num] = last_stat['norm']
                    except:
                        pass

            # Extract layer indices for decoder blocks
            for name, stats_list in self.grad_stats.items():
                if 'decoder_blocks' in name:
                    try:
                        layer_num = int(''.join(filter(str.isdigit, name.split('decoder_blocks.')[1].split('.')[0])))
                        if stats_list and len(stats_list) > 0:
                            last_stat = stats_list[-1]
                            decoder_layers[layer_num] = last_stat['norm']
                    except:
                        pass

            # Check for decreasing gradient norms with depth in encoder
            if len(encoder_layers) > 2:
                layer_nums = sorted(encoder_layers.keys())
                first_layer_norm = encoder_layers[layer_nums[0]]
                last_layer_norm = encoder_layers[layer_nums[-1]]

                if last_layer_norm < first_layer_norm * 0.1:
                    results["has_vanishing_gradients"] = True
                    results["layer_depth_issues"] = True
                    results["warnings"].append(
                        f"Gradient norm decreases significantly with encoder depth: "
                        f"first layer: {first_layer_norm:.8f}, last layer: {last_layer_norm:.8f}. "
                        f"Ratio: {last_layer_norm / first_layer_norm:.8f}. This indicates vanishing gradients."
                    )

            # Check for decreasing gradient norms with depth in decoder
            if len(decoder_layers) > 2:
                layer_nums = sorted(decoder_layers.keys())
                first_layer_norm = decoder_layers[layer_nums[0]]
                last_layer_norm = decoder_layers[layer_nums[-1]]

                if last_layer_norm < first_layer_norm * 0.1:
                    results["has_vanishing_gradients"] = True
                    results["layer_depth_issues"] = True
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

            # Write main verdict
            if vanishing_status["has_vanishing_gradients"]:
                f.write("VERDICT: VANISHING GRADIENTS DETECTED\n\n")
            else:
                f.write("VERDICT: NO CLEAR SIGN OF VANISHING GRADIENTS\n\n")

            # List all warnings and issues
            f.write("Warnings and Issues:\n")
            if vanishing_status["warnings"]:
                for i, warning in enumerate(vanishing_status["warnings"], 1):
                    f.write(f"{i}. {warning}\n")
            else:
                f.write("No specific issues detected.\n")

            f.write("\n")

            # Write direct gradient check information
            f.write("DIRECT GRADIENT CHECK:\n")
            f.write("-" * 50 + "\n")
            grad_count, total_params = self.check_gradients_directly()
            f.write(
                f"Parameters with gradients: {grad_count}/{total_params} ({grad_count / total_params * 100:.1f}%)\n\n")

            # Summarize gradient data statistics
            gradient_stats_file = os.path.join(self.output_dir, "gradient_stats.csv")
            if os.path.exists(gradient_stats_file):
                try:
                    grad_df = pd.read_csv(gradient_stats_file)
                    f.write("GRADIENT STATISTICS SUMMARY:\n")
                    f.write("-" * 50 + "\n")

                    if 'category' in grad_df.columns:
                        # Group by category
                        category_stats = grad_df.groupby('category')['norm'].agg(['mean', 'min', 'max'])
                        f.write("Gradient norms by category:\n")
                        for category, stats in category_stats.iterrows():
                            f.write(
                                f"  {category.ljust(12)}: mean={stats['mean']:.8f}, min={stats['min']:.8f}, max={stats['max']:.8f}\n")

                    # Overall statistics
                    f.write(f"\nOverall gradient statistics:\n")
                    f.write(f"  Mean gradient norm: {grad_df['norm'].mean():.8f}\n")
                    f.write(f"  Max gradient norm: {grad_df['norm'].max():.8f}\n")
                    f.write(f"  Min gradient norm: {grad_df['norm'].min():.8f}\n")
                    f.write(f"  Parameters with zero gradients: {grad_df['zeros'].mean() * 100:.2f}%\n")
                    f.write("\n")
                except Exception as e:
                    f.write(f"Error reading gradient statistics: {e}\n\n")

            # Write encoder-decoder ratio if available
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
                f.write("1. Consider using residual connections in problematic components\n")
                f.write("2. Try Layer Normalization before self-attention in transformer blocks\n")
                f.write("3. Adjust learning rates - use separate optimizer groups with different LRs\n")

                if vanishing_status["layer_depth_issues"]:
                    f.write("4. Reduce the depth of your model (especially in components showing gradient decay)\n")

                f.write("5. Use gradient clipping to stabilize training\n")
                f.write("6. Consider reducing the diversity loss weights initially in training\n")

                if vanishing_status["gradient_magnitude_issues"]:
                    f.write("7. Check your loss function scale and normalization\n")
                    f.write("8. Try different initialization strategies for model weights\n")
            else:
                f.write("Your model seems to have healthy gradient flow. If you're experiencing patch collapse,\n")
                f.write("the issue may be related to:\n")
                f.write("1. Loss function design or weighting\n")
                f.write("2. Initialization issues\n")
                f.write("3. Batch normalization or activation function problems\n")
                f.write("4. Dataset imbalance or preprocessing issues\n")

            # If there are missing gradients, add specific recommendations
            if grad_count < total_params * 0.5:
                f.write("\nMISSING GRADIENTS RECOMMENDATIONS:\n")
                f.write("1. Check for detached operations in forward pass that break the computation graph\n")
                f.write("2. Make sure gradient checkpointing is properly configured\n")
                f.write("3. Confirm all loss components are connected to the model parameters\n")
                f.write("4. Verify autocast (mixed precision) is properly implemented\n")
                f.write("5. Consider removing custom backward hooks that might interfere with gradient flow\n")

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


def run_gradient_diagnostics(model, train_loader, device, output_dir="gradient_diagnostics"):
    """
    Run a focused diagnostic session to check for vanishing gradients.
    Temporarily disables gradient checkpointing to ensure accurate gradient collection,
    then restores the original state before returning.
    """
    print("\n" + "=" * 80)
    print("RUNNING GRADIENT DIAGNOSTICS")
    print("=" * 80)

    # Create epoch-specific output directory
    epoch_dir = os.path.join(output_dir, f"epoch_0")
    os.makedirs(epoch_dir, exist_ok=True)

    # Temporarily disable gradient checkpointing for accurate diagnostics
    # by adding a flag attribute to the model
    setattr(model, '_disable_gradient_checkpointing', True)
    print("Gradient checkpointing temporarily disabled for diagnostics")

    # Create diagnostics tool
    diagnostics = GradientDiagnostics(model, output_dir=epoch_dir)

    # Register monitoring hooks - do this before optimizer creation
    diagnostics.register_gradient_hooks()
    diagnostics.register_activation_hooks()

    # Create optimizer and scaler (same settings as main training)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0001,
        weight_decay=0.05,
        betas=(0.9, 0.95)
    )
    scaler = torch.cuda.amp.GradScaler()

    # Keep track of batch idx for the diagnostics
    batch_idx = 0

    # Set model to train mode
    model.train()

    # Run a few batches with diagnostics
    print("Running diagnostic training on 3 batches...")
    for batch_idx, batch in enumerate(train_loader):
        # Only process a few batches
        if batch_idx >= 3:
            break

        # Move data to device
        hsi = batch['hsi'].to(device)
        aux_data = {k: v.to(device) if v is not None else None
                    for k, v in batch['aux_data'].items()}
        batch_idx_tensor = batch['batch_idx'].to(device)

        # Clear gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward pass with AMP
        with torch.cuda.amp.autocast():
            output = model(hsi, aux_data, batch_idx_tensor)
            loss = output['loss']

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Log diagnostics for this batch
        diagnostics.analyze_batch(batch_idx, loss=loss.item(), scaler=scaler)

        # Update weights with scaler
        scaler.step(optimizer)
        scaler.update()

        print(f"Processed diagnostic batch {batch_idx + 1}/3, Loss: {loss.item():.6f}")

    # Generate diagnostic report
    results = diagnostics.generate_report()

    # Print summary
    if results["has_vanishing_gradients"]:
        print("\n⚠️ VANISHING GRADIENTS DETECTED!")
        print("\nWarnings:")
        for warning in results["warnings"]:
            print(f"- {warning}")
    else:
        print("\n✅ No clear signs of vanishing gradients detected.")

    print(f"\nDetailed report available at: {epoch_dir}/summary_report.txt")

    # Re-enable gradient checkpointing by removing the flag
    if hasattr(model, '_disable_gradient_checkpointing'):
        delattr(model, '_disable_gradient_checkpointing')
    print("Gradient checkpointing re-enabled for normal training")

    print("=" * 80)

    return results