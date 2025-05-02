import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import json
from torch.cuda.amp import GradScaler
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import math


def visualize_model_structure(model, output_dir):
    """
    Create a comprehensive visualization of the model structure including:
    1. A hierarchical graph of module relationships using NetworkX without graphviz
    2. Parameter distribution across different layer categories
    3. Module dependency flow

    Args:
        model: The model to visualize
        output_dir: Directory to save visualizations

    Returns:
        str: Path to the saved visualization
    """
    import os
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from collections import defaultdict
    import json
    import networkx as nx
    from matplotlib.colors import LinearSegmentedColormap
    import math

    # Create output directory for structure visualizations
    structure_dir = os.path.join(output_dir, "model_structure")
    os.makedirs(structure_dir, exist_ok=True)

    # 1. Create a hierarchical graph of the model structure
    G = nx.DiGraph()

    # Track parameter counts
    param_counts = {}
    param_dimensions = {}
    trainable_counts = {}

    # Track module types
    module_types = {}

    # Function to recursively add modules to the graph
    def add_modules_to_graph(module, parent_name="model"):
        for name, child in module.named_children():
            # Create full module name
            full_name = f"{parent_name}.{name}" if parent_name else name

            # Add node to graph
            G.add_node(full_name)
            G.add_edge(parent_name, full_name)

            # Get module type
            module_type = child.__class__.__name__
            module_types[full_name] = module_type

            # Count parameters
            param_count = sum(p.numel() for p in child.parameters())
            trainable_count = sum(p.numel() for p in child.parameters() if p.requires_grad)

            param_counts[full_name] = param_count
            trainable_counts[full_name] = trainable_count

            # Store sample parameter dimensions for visualization
            dims = []
            for name, param in child.named_parameters():
                if param.requires_grad:
                    dims.append(f"{name}: {tuple(param.shape)}")
            param_dimensions[full_name] = dims

            # Recurse for children
            add_modules_to_graph(child, full_name)

    # Start with the model itself
    G.add_node("model")
    add_modules_to_graph(model)

    # 2. Draw hierarchical graph using NetworkX's built-in layout options
    plt.figure(figsize=(20, 15))

    # Replace graphviz with networkx's hierarchical layout
    # First try to use a hierarchical layout if available
    try:
        # For newer NetworkX versions (2.6+)
        pos = nx.nx_pydot.pydot_layout(G, prog="dot")
    except:
        try:
            # If pydot fails, try pygraphviz variant (should never happen if graphviz is issue)
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except:
            # Fall back to a pure networkx implementation that doesn't require external deps
            # Use a combination of spring_layout with iterations and fixed y-positions based on depth

            # Calculate node depths
            def get_node_depth(node):
                if node == "model":
                    return 0
                parent_depths = [get_node_depth(parent) for parent in G.predecessors(node)]
                return 1 + max(parent_depths) if parent_depths else 0

            node_depths = {node: get_node_depth(node) for node in G.nodes()}
            max_depth = max(node_depths.values())

            # Create initial layout with spring model
            pos = nx.spring_layout(G, k=1.5 / math.sqrt(len(G.nodes())), iterations=50, seed=42)

            # Adjust y-positions based on depth and apply some horizontal spreading
            for node, (x, y) in pos.items():
                depth = node_depths[node]
                # Normalize depth for y-position (top to bottom)
                y_pos = 1.0 - (depth / max(max_depth, 1))
                # Keep original x but normalize it a bit
                x_pos = x * 0.8
                pos[node] = (x_pos, y_pos)

    # Color nodes by parameter count using a colormap
    if param_counts:
        max_count = max(param_counts.values())
        min_count = min([count for count in param_counts.values() if count > 0]) if any(
            count > 0 for count in param_counts.values()) else 0

        # Create custom colormap from blue to red
        cmap = LinearSegmentedColormap.from_list("param_count", ["#E0F7FF", "#FF5733"])

        # Calculate colors based on log scale for better visualization
        node_colors = []
        for node in G.nodes():
            if node in param_counts and param_counts[node] > 0:
                # Use log scale to better visualize parameter counts
                log_count = math.log(param_counts[node]) if param_counts[node] > 0 else 0
                log_min = math.log(min_count) if min_count > 0 else 0
                log_max = math.log(max_count) if max_count > 0 else 1

                if log_max > log_min:
                    color_val = (log_count - log_min) / (log_max - log_min)
                else:
                    color_val = 0.5

                color = cmap(color_val)
            else:
                color = "#EEEEEE"  # Light gray for nodes without parameters

            node_colors.append(color)

        # Node size based on parameter count (log scale for better visualization)
        node_sizes = []
        for node in G.nodes():
            if node in param_counts and param_counts[node] > 0:
                size = 300 * (math.log(param_counts[node] + 1) / math.log(max_count + 1))
                node_sizes.append(max(50, size))
            else:
                node_sizes.append(50)  # Default size for nodes without parameters
    else:
        node_colors = ["#EEEEEE"] * len(G.nodes())
        node_sizes = [50] * len(G.nodes())

    # Draw the network with improved labels
    nx.draw(G, pos,
            with_labels=True,
            node_color=node_colors,
            node_size=node_sizes,
            font_size=8,
            font_weight="bold",
            edge_color="#CCCCCC",
            width=1.0,
            arrows=True)

    # Improve readability by adding node labels with more details
    labels = {}
    for node in G.nodes():
        if node in param_counts:
            labels[node] = f"{node.split('.')[-1]}\n{param_counts[node]:,}"
        else:
            labels[node] = node.split('.')[-1]

    # Add a legend for node sizes/colors
    if param_counts:
        plt.figtext(0.01, 0.01, "Node size and color represent parameter count",
                    fontsize=10, ha='left')

    plt.title("Model Structure Hierarchy", fontsize=16)
    plt.tight_layout()
    model_structure_path = os.path.join(structure_dir, "model_hierarchy.png")
    plt.savefig(model_structure_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Create parameter distribution pie chart
    category_params = {}
    layer_categories = {
        "encoder": ["patch_embed", "pos_embed", "blocks", "cross_attn", "norm"],
        "decoder": ["decoder_embed", "decoder_blocks", "decoder_norm", "decoder_pred", "decoder_pos_embed"],
        "projections": ["proj_head_global", "proj_head_spatial", "modality_proj"],
        "aux_encoders": ["aux_encoder"],
        "contrastive": ["temperature", "mask_token"]
    }

    # Categorize parameters
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            category = "other"
            for cat_name, keywords in layer_categories.items():
                if any(keyword in param_name for keyword in keywords):
                    category = cat_name
                    break

            if category not in category_params:
                category_params[category] = 0

            category_params[category] += param.numel()

    # Create pie chart
    plt.figure(figsize=(12, 8))

    labels = []
    sizes = []
    explode = []
    colors = plt.cm.tab10(np.linspace(0, 1, len(category_params)))

    for i, (category, count) in enumerate(sorted(category_params.items(), key=lambda x: x[1], reverse=True)):
        percentage = count / sum(category_params.values()) * 100
        labels.append(f"{category}: {percentage:.1f}%\n({count:,} params)")
        sizes.append(count)
        explode.append(0.1 if i == 0 else 0)

    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=140, textprops={'fontsize': 9})
    plt.axis('equal')
    plt.title("Parameter Distribution by Category", fontsize=16)
    plt.tight_layout()

    pie_chart_path = os.path.join(structure_dir, "parameter_distribution.png")
    plt.savefig(pie_chart_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Create detailed structure report with layer information
    report_path = os.path.join(structure_dir, "model_structure_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:  # Added explicit encoding to handle Unicode
        f.write("=" * 80 + "\n")
        f.write("MODEL STRUCTURE REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Overall statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,} ({trainable_params / total_params * 100:.1f}%)\n\n")

        # Write parameter distribution by category
        f.write("Parameter Distribution by Category:\n")
        f.write("-" * 50 + "\n")

        for category, count in sorted(category_params.items(), key=lambda x: x[1], reverse=True):
            percentage = count / trainable_params * 100
            f.write(f"{category.ljust(15)}: {count:,} ({percentage:.2f}%)\n")

        f.write("\n")

        # Write detailed layer information
        f.write("Detailed Layer Structure:\n")
        f.write("-" * 50 + "\n")

        def write_module_info(module, prefix="", depth=0):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                module_type = child.__class__.__name__

                param_count = sum(p.numel() for p in child.parameters())
                trainable_count = sum(p.numel() for p in child.parameters() if p.requires_grad)

                indent = "  " * depth
                # Changed bullet character from Unicode (●) to ASCII (*)
                f.write(f"{indent}* {name} ({module_type}): {trainable_count:,}/{param_count:,} params\n")

                # Print parameter shapes for leaf modules
                if not list(child.named_children()) and list(child.named_parameters()):
                    for param_name, param in child.named_parameters():
                        if param.requires_grad:
                            indent_params = "  " * (depth + 1)
                            f.write(f"{indent_params}- {param_name}: {tuple(param.shape)}\n")

                write_module_info(child, full_name, depth + 1)

        write_module_info(model)

    # 5. Create a flow diagram of main components
    # This is a simplified directed graph showing main model components
    H = nx.DiGraph()

    # Define main functional components
    main_components = [
        "Encoder", "Decoder", "Aux Encoders", "Cross-Attention",
        "Contrastive Learning", "MAE Reconstruction"
    ]

    # Define relationships
    edges = [
        ("Input", "Encoder"),
        ("Input", "Aux Encoders"),
        ("Encoder", "Cross-Attention"),
        ("Aux Encoders", "Cross-Attention"),
        ("Cross-Attention", "Decoder"),
        ("Encoder", "Contrastive Learning"),
        ("Aux Encoders", "Contrastive Learning"),
        ("Decoder", "MAE Reconstruction"),
        ("MAE Reconstruction", "Output Loss"),
        ("Contrastive Learning", "Output Loss")
    ]

    # Add nodes and edges
    H.add_nodes_from(["Input", "Output Loss"] + main_components)
    H.add_edges_from(edges)

    # Create component flow diagram
    plt.figure(figsize=(14, 10))

    # Position nodes in a more logical flow - without graphviz
    # Define fixed positions manually for a clear flow layout
    pos = {
        "Input": (0, 0),
        "Encoder": (1, 1),
        "Aux Encoders": (1, -1),
        "Cross-Attention": (2, 0),
        "Decoder": (3, 0),
        "MAE Reconstruction": (4, 0.5),
        "Contrastive Learning": (4, -0.5),
        "Output Loss": (5, 0)
    }

    # Custom node colors
    node_colors = {
        "Input": "#E0F7FF",
        "Encoder": "#FFD966",
        "Aux Encoders": "#A9D18E",
        "Cross-Attention": "#C5A5CF",
        "Decoder": "#FF9966",
        "MAE Reconstruction": "#9FC5F8",
        "Contrastive Learning": "#F8C4B4",
        "Output Loss": "#EA9999"
    }

    # Draw the network
    node_color_list = [node_colors.get(node, "#EEEEEE") for node in H.nodes()]

    nx.draw(H, pos,
            with_labels=True,
            node_color=node_color_list,
            node_size=2500,
            font_size=10,
            font_weight="bold",
            edge_color="#666666",
            width=2.0,
            arrowsize=20,
            arrows=True)

    plt.title("Model Functional Flow", fontsize=16)
    plt.tight_layout()
    flow_diagram_path = os.path.join(structure_dir, "model_flow_diagram.png")
    plt.savefig(flow_diagram_path, dpi=300, bbox_inches="tight")
    plt.close()

    return structure_dir

class GradientDiagnostics:
    """
    An improved diagnostics tool to monitor and analyze gradients in the MultiModalSpectralGPT model
    to help identify vanishing gradient problems across the entire model.

    Key improvements:
    - Direct gradient collection after backward pass instead of hooks
    - Better handling of mixed precision training
    - Enhanced visualization and analysis
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

    def visualize_model_structure(self):
        """
        Visualize the model's structure and parameter distribution with accurate categorization.

        Returns:
            str: Path to the saved visualizations
        """
        structure_dir = visualize_model_structure(self.model, self.output_dir, layer_categories=self.layer_categories)
        self.log(f"Model structure visualizations saved to: {structure_dir}")
        return structure_dir

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
        """
        This method is kept for backward compatibility, but we're no longer using hooks
        for gradient collection as they don't work reliably with mixed precision training.
        """
        self.log("Gradient hooks are no longer used - gradients will be collected directly after backward()")
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

    def collect_and_analyze_gradients(self, batch_idx, loss=None, scaler=None):
        """
        Collect and analyze gradients after backward() but before optimizer step.
        This is a key improvement to correctly capture gradients when using mixed precision training.

        Args:
            batch_idx: Current batch index
            loss: Loss value for the current batch
            scaler: GradScaler instance for mixed precision training
        """
        self.batch_counter = batch_idx
        self.log(f"Analyzing batch {batch_idx}")

        # Track GradScaler if provided
        if scaler is not None:
            self.track_gradscaler(scaler)

        # Collect gradients directly from parameters
        grad_count = 0
        total_params = 0
        grad_by_category = defaultdict(int)
        interesting_grads = []

        for name, param in self.named_parameters:
            if param.requires_grad:
                total_params += 1
                if param.grad is not None:
                    grad_count += 1

                    # Calculate gradient statistics
                    with torch.no_grad():
                        grad = param.grad
                        grad_abs = grad.abs()

                        # Handle different tensor shapes
                        if grad.dim() > 1:
                            flat_grad = grad.view(-1)
                            median_val = torch.median(flat_grad)
                        else:
                            median_val = torch.median(grad)

                        # Collect important stats
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

                        # Categorize the parameter with improved logic
                        category = "other"

                        # Match parameters to categories with improved logic
                        if any(keyword in name for keyword in self.layer_categories["decoder"]):
                            category = "decoder"
                        elif name.startswith(
                                "blocks.") or name == "norm.weight" or name == "norm.bias" or "patch_embed" in name or "pos_embed" in name:
                            category = "encoder"
                        elif any(keyword in name for keyword in self.layer_categories["projections"]):
                            category = "projections"
                        elif any(keyword in name for keyword in self.layer_categories["aux_encoders"]):
                            category = "aux_encoders"
                        elif any(keyword in name for keyword in self.layer_categories["contrastive"]):
                            category = "contrastive"

                        stats["category"] = category
                        grad_by_category[category] += 1

                        # Store the stats
                        self.grad_stats[name].append(stats)

                        # Track interesting gradients (key layers) for reporting
                        if 'decoder_pred' in name or 'blocks.0' in name or 'decoder_blocks.0' in name:
                            grad_mean = grad_abs.mean().item()
                            interesting_grads.append(f"{name}: {grad_mean:.8f}")

        # Log detailed gradient information
        self.log(f"Collected gradients for {grad_count}/{total_params} parameters")

        # Log categories with gradients
        for category, count in grad_by_category.items():
            self.log(f"  - {category}: {count} parameters have gradients")

        # Log sample interesting gradients
        if interesting_grads:
            self.log(f"Sample gradients: {', '.join(interesting_grads[:5])}")
        else:
            self.log(f"No interesting gradients found")

        # Check for gradient coverage across categories
        missing_categories = []
        for category in self.layer_categories:
            if grad_by_category[category] == 0:
                missing_categories.append(category)

        if missing_categories:
            self.log(f"Warning: No gradients collected for categories: {missing_categories}")
        else:
            self.log(f"Good gradient coverage: gradients found for all categories")

        # Return self for chaining
        return self

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
        """
        Backwards compatibility method that redirects to collect_and_analyze_gradients.
        """
        self.log("analyze_batch is deprecated, redirecting to collect_and_analyze_gradients")
        return self.collect_and_analyze_gradients(batch_idx, loss, scaler)

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
        """Generate a comprehensive diagnostic report with added model structure visualization"""
        self.log("\n" + "=" * 50)
        self.log("Generating Gradient Diagnostics Report")
        self.log("=" * 50)

        # 1. Visualize model structure before other diagnostics
        self.log("\nVisualizing model structure...")
        structure_dir = self.visualize_model_structure()
        self.log(f"Model structure visualizations saved to: {structure_dir}")

        # 2. Save all collected statistics to CSV
        self._save_stats_to_csv()

        # 3. Generate plots
        self._generate_plots()

        # 4. Check for gradient vanishing signs
        vanishing_status = self._check_for_vanishing_gradients()

        # 5. Summarize findings in the report
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
        """Write a summary report based on the analysis with added model structure reference"""
        summary_file = os.path.join(self.output_dir, "summary_report.txt")

        with open(summary_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("GRADIENT DIAGNOSTICS SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Reference to model structure visualization
            f.write("MODEL STRUCTURE ANALYSIS:\n")
            f.write("-" * 50 + "\n")
            structure_dir = os.path.join(self.output_dir, "model_structure")
            f.write(f"Detailed model structure visualizations available at: {structure_dir}\n")
            f.write("Key visualizations include:\n")
            f.write("  - model_hierarchy.png: Hierarchical graph of model components\n")
            f.write("  - parameter_distribution.png: Parameter distribution by category\n")
            f.write("  - model_flow_diagram.png: Functional flow of model components\n")
            f.write("  - model_structure_report.txt: Detailed text report of model structure\n\n")

            # Write main vanishing gradients verdict
            f.write("VANISHING GRADIENTS ANALYSIS:\n")
            f.write("-" * 50 + "\n")

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
            f.write(f"Model structure analysis: {os.path.join(self.output_dir, 'model_structure')}\n")

        # Also save a JSON version of the analysis
        with open(os.path.join(self.output_dir, "vanishing_gradient_analysis.json"), "w") as f:
            json.dump(vanishing_status, f, indent=2)

        return summary_file


def run_gradient_diagnostics(model, train_loader, device, output_dir="gradient_diagnostics"):
    """
    Run a focused diagnostic session to check for vanishing gradients.
    Temporarily disables gradient checkpointing to ensure accurate gradient collection,
    then restores the original state before returning.
    Uses a smaller batch size to avoid memory errors.

    Key improvement: Collects gradients after backward() but before optimizer step.
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

    # We'll only register activation hooks - we'll collect gradients directly later
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

    # Use a smaller batch size for diagnostics
    diagnostic_batch_size = 1  # Use smallest possible batch size

    print(f"Running diagnostic training with reduced batch size of {diagnostic_batch_size}...")

    # Run a few batches with diagnostics
    for _, batch in enumerate(train_loader):
        # Only process a few batches
        if batch_idx >= 3:
            break

        # Create sub-batches to reduce memory usage
        sub_batches = create_sub_batches(batch, diagnostic_batch_size)

        for sub_batch in sub_batches:
            # Move data to device
            hsi = sub_batch['hsi'].to(device)
            aux_data = {k: v.to(device) if v is not None else None
                        for k, v in sub_batch['aux_data'].items()}
            batch_idx_tensor = sub_batch['batch_idx'].to(device)

            # Clear gradients
            optimizer.zero_grad(set_to_none=True)

            # Forward pass with AMP
            with torch.cuda.amp.autocast():
                output = model(hsi, aux_data, batch_idx_tensor)
                loss = output['loss']

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # IMPORTANT: This is the key change - collect gradients AFTER backward()
            # but BEFORE optimizer step and scaler update
            diagnostics.collect_and_analyze_gradients(batch_idx, loss=loss.item(), scaler=scaler)

            # Update weights with scaler
            scaler.step(optimizer)
            scaler.update()

            print(f"Processed diagnostic sub-batch {batch_idx + 1}/3, Loss: {loss.item():.6f}")

            # Clear memory
            del hsi, aux_data, batch_idx_tensor, output, loss
            torch.cuda.empty_cache()

            batch_idx += 1
            if batch_idx >= 3:
                break

    # Generate diagnostic report
    results = diagnostics.generate_report()

    # Print summary
    if results[0]["has_vanishing_gradients"]:
        print("\n⚠️ VANISHING GRADIENTS DETECTED!")
        print("\nWarnings:")
        for warning in results[0]["warnings"]:
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


def create_sub_batches(batch, sub_batch_size):
    """
    Split a large batch into smaller sub-batches to reduce memory usage.

    Args:
        batch: Original batch data
        sub_batch_size: Size of each sub-batch

    Returns:
        List of sub-batches
    """
    original_batch_size = batch['hsi'].shape[0]
    sub_batches = []

    for i in range(0, original_batch_size, sub_batch_size):
        end_idx = min(i + sub_batch_size, original_batch_size)
        sub_batch = {
            'hsi': batch['hsi'][i:end_idx],
            'aux_data': {
                k: (v[i:end_idx] if v is not None else None)
                for k, v in batch['aux_data'].items()
            },
            'batch_idx': batch['batch_idx'][i:end_idx]
        }

        # Add thickness mask if available
        if 'thickness_mask' in batch and batch['thickness_mask'] is not None:
            sub_batch['thickness_mask'] = batch['thickness_mask'][i:end_idx]

        sub_batches.append(sub_batch)

    return sub_batches