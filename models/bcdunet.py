import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell implementation for BCDUNet"""
    
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super(ConvLSTMCell, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # Gates: input, forget, cell, output
        self.gates = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=True
        )
        
        # Initialize weights using Kaiming initialization
        nn.init.kaiming_normal_(self.gates.weight, mode='fan_in', nonlinearity='sigmoid')
        
    def forward(self, x, h_prev=None, c_prev=None):
        # Get batch and spatial sizes
        batch_size, _, height, width = x.size()
        
        # Initialize hidden and cell states if not provided
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
        if c_prev is None:
            c_prev = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
            
        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=1)
        
        # Calculate gates
        gates = self.gates(combined)
        
        # Split gates into their components
        input_gate, forget_gate, cell_gate, output_gate = torch.split(gates, self.hidden_channels, dim=1)
        
        # Apply activations to gates
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        output_gate = torch.sigmoid(output_gate)
        
        # Update cell state and hidden state
        c_next = forget_gate * c_prev + input_gate * cell_gate
        h_next = output_gate * torch.tanh(c_next)
        
        return h_next, c_next


class BCDUNet(nn.Module):
    """PyTorch implementation of BCDUNet with dense blocks"""
    
    def __init__(self, config=None):
        super(BCDUNet, self).__init__()
        
        if config is None:
            config = {}
            
        # Parameters from config
        self.in_channels = config.get("in_channels", 3)
        self.out_channels = config.get("out_channels", 1)  # Default for vessel segmentation
        self.block_depth = config.get("block_depth", 3)  # D3 or D1
        self.filters = config.get("filters", 64)  # Starting filter count
        self.dropout_rate = config.get("dropout_rate", 0.5)
        
        # Print model configuration
        print(f"Initialized BCDUNet model with {self.in_channels} input channels, "
              f"{self.out_channels} output channels, block depth: {self.block_depth}")
        
        # Initial Conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.filters, self.filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # Encoder Path
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.filters, self.filters*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.filters*2, self.filters*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.filters*2, self.filters*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.filters*4, self.filters*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.dropout_rate)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # Bridge / Dense connectivity
        # D1 block
        self.bridge_conv1 = nn.Sequential(
            nn.Conv2d(self.filters*4, self.filters*8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.filters*8, self.filters*8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.dropout_rate)
        )
        
        # D2 block (if using D3)
        if self.block_depth == 3:
            self.bridge_conv2 = nn.Sequential(
                nn.Conv2d(self.filters*8, self.filters*8, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.filters*8, self.filters*8, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(self.dropout_rate)
            )
            
            # D3 block
            self.bridge_conv3 = nn.Sequential(
                nn.Conv2d(self.filters*16, self.filters*8, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.filters*8, self.filters*8, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(self.dropout_rate)
            )
        
        # Decoder Path
        self.up6 = nn.Sequential(
            nn.ConvTranspose2d(self.filters*8, self.filters*4, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.filters*4),
            nn.ReLU(inplace=True)
        )
        
        # ConvLSTM replacement for level 1
        self.lstm1_cell = ConvLSTMCell(
            input_channels=self.filters*4,
            hidden_channels=self.filters*4,
            kernel_size=3
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(self.filters*4, self.filters*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.filters*4, self.filters*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.up7 = nn.Sequential(
            nn.ConvTranspose2d(self.filters*4, self.filters*2, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.filters*2),
            nn.ReLU(inplace=True)
        )
        
        # ConvLSTM replacement for level 2
        self.lstm2_cell = ConvLSTMCell(
            input_channels=self.filters*2,
            hidden_channels=self.filters*2,
            kernel_size=3
        )
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(self.filters*2, self.filters*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.filters*2, self.filters*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(self.filters*2, self.filters, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.filters),
            nn.ReLU(inplace=True)
        )
        
        # ConvLSTM replacement for level 3
        self.lstm3_cell = ConvLSTMCell(
            input_channels=self.filters,
            hidden_channels=self.filters,
            kernel_size=3
        )
        
        self.conv8 = nn.Sequential(
            nn.Conv2d(self.filters, self.filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.filters, self.filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Final output convolutions
        self.conv9 = nn.Conv2d(self.filters, 2, kernel_size=3, padding=1)
        self.relu9 = nn.ReLU(inplace=True)
        self.conv10 = nn.Conv2d(2, self.out_channels, kernel_size=1)
        
        # Sigmoid activation for binary segmentation
        self.sigmoid = nn.Sigmoid()
        
        # Initialize all weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # Encoder path
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        
        # Bridge with dense connectivity based on block depth
        conv4 = self.bridge_conv1(pool3)
        
        if self.block_depth == 3:
            # D2 block
            conv4_2 = self.bridge_conv2(conv4)
            
            # Dense connection from D1 to D3
            merge_dense = torch.cat([conv4_2, conv4], dim=1)
            
            # D3 block
            conv4_3 = self.bridge_conv3(merge_dense)
            bridge_output = conv4_3
        else:
            bridge_output = conv4
        
        # Decoder path
        up6 = self.up6(bridge_output)
        
        # Apply ConvLSTM for level 1 - simplified approach using our custom cell
        # Initialize the hidden states with the skip connection
        h6, _ = self.lstm1_cell(up6, conv3, None)
        conv6 = self.conv6(h6)
        
        # Continue decoder path
        up7 = self.up7(conv6)
        
        # Apply ConvLSTM for level 2
        h7, _ = self.lstm2_cell(up7, conv2, None)
        conv7 = self.conv7(h7)
        
        # Final decoder block
        up8 = self.up8(conv7)
        
        # Apply ConvLSTM for level 3
        h8, _ = self.lstm3_cell(up8, conv1, None)
        conv8 = self.conv8(h8)
        
        # Final convolutions
        conv9 = self.relu9(self.conv9(conv8))
        conv10 = self.conv10(conv9)
        
        # Apply sigmoid for binary segmentation
        if self.out_channels == 1:
            output = self.sigmoid(conv10)
        else:
            # No sigmoid for multi-class segmentation
            output = conv10
            
        return output


class BCDUNetModel(nn.Module):
    """BCDUNet model wrapped for compatibility with the training framework"""
    
    @staticmethod
    def get_default_config(image_size):
        """
        Get default configuration for BCDUNet model.
        
        Args:
            image_size: Size of input image (for consistent interface)
            
        Returns:
            Dictionary with default configuration
        """
        return {
            "in_channels": 3,
            "out_channels": 1,  # Default for vessel segmentation
            "block_depth": 3,   # Default to D3 version
            "filters": 64,      # Starting filter count
            "dropout_rate": 0.5,
            "task": "vessel"    # Default task
        }
    
    def __init__(self, 
                 in_channels=3, 
                 out_channels=1, 
                 block_depth=3, 
                 filters=64, 
                 dropout_rate=0.5, 
                 task="vessel",
                 **kwargs):  # Accept additional params for compatibility
        """
        Initialize the BCDUNet model with individual parameters.
        This matches the initialization pattern used by the training pipeline.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (1 for binary segmentation)
            block_depth: Depth of the dense block (1 or 3)
            filters: Starting filter count (64 in original paper)
            dropout_rate: Dropout probability
            task: Task type ('vessel' or 'odoc')
            **kwargs: Additional parameters for compatibility
        """
        super(BCDUNetModel, self).__init__()
        
        # Adjust outputs based on task
        if task == "odoc":
            out_channels = 3  # ODOC has 3 output classes
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_depth = block_depth
        self.filters = filters
        self.dropout_rate = dropout_rate
        self.task = task
        
        # Print model configuration
        print(f"Initialized BCDUNet model configuration with {self.in_channels} input channels, "
              f"{self.out_channels} output channels, block depth: {self.block_depth}")
        
        # Build the model immediately
        config = {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "block_depth": self.block_depth,
            "filters": self.filters,
            "dropout_rate": self.dropout_rate,
            "task": self.task
        }
        
        # Create and store the BCDUNet model
        self.model = BCDUNet(config)
    
    def build(self):
        """
        Build the BCDUNet model with the specified configuration.
        
        Returns:
            A configured BCDUNet model instance
        """
        return self.model
    
    def forward(self, x):
        return self.model(x)