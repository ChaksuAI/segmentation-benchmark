"""
Utility modules for MONAI training and inference
"""

from .data import (
    get_datasets,
    get_data_loaders,
    get_combined_datasets,
    get_test_dataset
)

from .loss import get_loss_function

from .metrics import (
    get_metric,
    calculate_per_class_metrics,
    calculate_metrics,
    save_evaluation_results
)

from .transforms import (
    ConvertSegmentationMaskd,
    get_train_transforms,
    get_val_transforms,
    get_test_transforms
)

from .cli import (
    print_header,
    print_config,
    parse_args
)

from .io import (
    create_output_directories,
    create_output_directory,
    save_config,
    get_dataset_path
)

from .model import (
    generate_default_config,
    load_model,
    save_model,
    create_symlink,
    initialize_model
)

from .visualization import (
    save_segmentation_result,
    plot_training_curves,
    log_epoch_metrics
)

__all__ = [
    # Data utilities
    "get_datasets", "get_data_loaders","get_combined_datasets", "get_test_dataset",
    # Loss functions
    "get_loss_function",
    # Metrics
    "get_metric", "calculate_per_class_metrics", "calculate_metrics", "save_evaluation_results",
    # Transforms
    "ConvertSegmentationMaskd", "get_train_transforms", "get_val_transforms", "get_test_transforms",
    # CLI utilities
    "print_header", "print_config", "parse_args",
    # I/O utilities
    "create_output_directories", "create_output_directory", "save_config", "get_dataset_path",
    # Model utilities
    "generate_default_config", "load_model", "save_model", "create_symlink", "initialize_model",
    # Visualization utilities
    "save_segmentation_result", "plot_training_curves", "log_epoch_metrics"
]
