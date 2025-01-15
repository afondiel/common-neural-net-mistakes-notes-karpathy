# Common Neural Network Mistakes and Best Practices Notes (From Andrej Karpathy [Tweet](https://x.com/karpathy/status/1013244313327681536))

## Oveview

As I continue my endless journey of deep learning research & exploration, I discovered Andrej Karpathy's insightful [recipe post](https://karpathy.github.io/2019/04/25/recipe/) and traced it back to his original [tweet thread](https://x.com/karpathy/status/1013244313327681536) from 2018. To deepen my understanding, I asked [Claude](https://claude.ai) to expand these insights with practical examples and solutions.

> [!IMPORTANT]  
> These notes may have some variations compared to Karpathy's extended [version](https://karpathy.github.io/2019/04/25/recipe/). So, I recommend you to read it for a comprehensive understanding.

## Usage
I curated this guide as both a "personal reminder" and a practical reference to help me approach deep learning problems more efficiently.

## Table of Contents
- [Overview](#overview)
- [Usage](#motivation)
- [1. Overfit Single Batch First](#overfit-single-batch-first)
- [2. Train/Eval Mode Toggle](#traineval-mode-toggle)
- [3. Gradient Accumulation Reset](#gradient-accumulation-reset)
- [4. Loss Function Input Format](#loss-function-input-format)
- [5. BatchNorm and Bias](#batchnorm-and-bias)
- [6. Tensor Reshaping Operations](#tensor-reshaping-operations)
[References](#references)

## 1. Overfit Single Batch First

### The Problem
When training neural networks, it's crucial to verify that your model can learn at all before trying to generalize to the full dataset. Many practitioners skip this step and waste time debugging on the full dataset.

### The Solution
Always start by overfitting a single batch. This verifies that:
1. Your model has sufficient capacity to learn
2. Your loss function is properly implemented
3. Your optimization setup works

```python
def verify_model_learns(model, loss_fn, optimizer):
    # Get a single batch of data
    x_batch, y_batch = next(iter(train_loader))
    
    # Training loop on single batch
    for epoch in range(100):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(x_batch)
        loss = loss_fn(outputs, y_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Check if model is learning
        if loss.item() < 0.01:
            print("Success! Model can learn on single batch")
            return True
            
    print("Warning: Model failed to learn single batch")
    return False

# Usage example
model = MyNeuralNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

verify_model_learns(model, loss_fn, optimizer)

# Expected output for a properly working model:
"""
Epoch 0, Loss: 2.3046
Epoch 10, Loss: 1.8721
Epoch 20, Loss: 1.2154
Epoch 30, Loss: 0.6832
Epoch 40, Loss: 0.2547
Epoch 50, Loss: 0.0892
Epoch 60, Loss: 0.0075
Success! Model can learn on single batch
"""

# Expected output for a problematic model:
"""
Epoch 0, Loss: 2.3046
Epoch 10, Loss: 2.2998
Epoch 20, Loss: 2.2987
Epoch 30, Loss: 2.2982
...
Warning: Model failed to learn single batch
"""

# Common issues if model fails to learn:
# 1. Learning rate too high/low
# 2. Incorrect loss function
# 3. Model architecture issues
# 4. Data normalization problems
```

## 2. Train/Eval Mode Toggle

### The Problem
Many layers like Dropout and BatchNorm behave differently during training and evaluation. Forgetting to toggle these modes leads to inconsistent results.

### The Solution
Always explicitly set the mode before training and evaluation phases:

```python
def training_loop(model, train_loader, val_loader, epochs):
    for epoch in range(epochs):
        # Training phase
        model.train()  # Set train mode
        for batch in train_loader:
            # Training steps
            pass
            
        # Validation phase
        model.eval()  # Set evaluation mode
        with torch.no_grad():  # Disable gradient computation
            for batch in val_loader:
                # Validation steps
                pass

# Best practice: Create wrapper functions
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for inputs, targets in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return total_loss / len(loader), 100. * correct / total

# Example usage and expected output:
"""
# During training:
train_loss = train_epoch(model, train_loader, criterion, optimizer)
print(f'Training Loss: {train_loss:.3f}')
# Output: Training Loss: 0.342

# During evaluation:
val_loss, val_acc = evaluate(model, val_loader, criterion)
print(f'Validation Loss: {val_loss:.3f}, Accuracy: {val_acc:.2f}%')
# Output: Validation Loss: 0.285, Accuracy: 91.45%

# Key differences between train/eval mode:
# 1. Dropout layers: Active in train(), inactive in eval()
# 2. BatchNorm: Updates statistics in train(), uses running stats in eval()
# 3. Gradient computation: Enabled in train(), disabled in eval()
"""
```

## 3. Gradient Accumulation Reset

### The Problem
PyTorch accumulates gradients by default. Not zeroing gradients before backward pass leads to incorrect gradient updates.

### The Solution
Always clear gradients before computing new ones:

```python
def train_step(model, optimizer, loss_fn, x, y):
    # Clear gradients before new backward pass
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(x)
    loss = loss_fn(outputs, y)
    
    # Backward pass
    loss.backward()
    
    # Update weights
    optimizer.step()
    
    return loss.item()

# Example usage and output:
"""
# Single training step
loss = train_step(model, optimizer, loss_fn, inputs, targets)
print(f'Step Loss: {loss:.4f}')
# Output: Step Loss: 0.3254

# What happens if you forget zero_grad():
# Iteration 1: Loss: 0.3254 (correct)
# Iteration 2: Loss: 0.8976 (incorrect - gradients accumulated!)
# Iteration 3: Loss: 1.2543 (even worse!)
"""

# Alternative: Zero gradients at the start of each batch
for batch_idx, (inputs, targets) in enumerate(train_loader):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    
    if batch_idx % 100 == 0:
        print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')

# Expected output:
"""
Batch 0, Loss: 2.3046
Batch 100, Loss: 1.8721
Batch 200, Loss: 1.2154
Batch 300, Loss: 0.6832

# Common gradient-related issues:
1. Exploding gradients (loss increases rapidly)
2. Vanishing gradients (loss barely changes)
3. NaN losses (gradients became invalid)
"""
```

## 4. Loss Function Input Format

### The Problem
Many loss functions expect raw logits (pre-softmax values), but practitioners often mistakenly apply softmax first.

### The Solution
Understand your loss function's expected input format:

```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # Output raw logits
        )
    
    def forward(self, x):
        return self.layers(x)  # Don't apply softmax here

# CrossEntropyLoss applies log_softmax internally
criterion = nn.CrossEntropyLoss()

# For inference when you need probabilities
def predict_proba(model, x):
    logits = model(x)
    return F.softmax(logits, dim=1)

# Common mistake example (DON'T DO THIS):
def incorrect_training():
    outputs = model(x)
    probs = F.softmax(outputs, dim=1)  # Wrong! Don't apply softmax
    loss = criterion(probs, targets)    # CrossEntropyLoss expects logits
```

## 5. BatchNorm and Bias

### The Problem
Using bias in layers before BatchNorm is redundant because BatchNorm includes a learnable shift parameter (β).

### The Solution
Disable bias for layers followed by BatchNorm, but keep it for the output layer:

```python
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Correct usage: bias=False before BatchNorm
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Correct usage: bias=True for output layer
        self.fc = nn.Linear(64, 10, bias=True)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        return self.fc(x)

# Helper function to check model for proper bias usage
def check_bias_usage(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            next_module = None
            # Get next module (if exists)
            try:
                layers = list(model.modules())
                idx = layers.index(module)
                if idx + 1 < len(layers):
                    next_module = layers[idx + 1]
            except:
                pass
            
            if module.bias is not None and isinstance(next_module, nn.BatchNorm2d):
                print(f"Warning: {name} has bias=True before BatchNorm")
            elif module.bias is None and not isinstance(next_module, nn.BatchNorm2d):
                print(f"Warning: {name} has bias=False but no BatchNorm follows")

# Example usage and output:
"""
# Example 1: Correct model architecture
model = nn.Sequential(
    nn.Conv2d(3, 64, 3, bias=False),  # Correct: no bias before BatchNorm
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Linear(64, 10, bias=True)      # Correct: bias in final layer
)
check_bias_usage(model)
# Output: (no warnings)

# Example 2: Incorrect model architecture
model = nn.Sequential(
    nn.Conv2d(3, 64, 3, bias=True),   # Wrong: unnecessary bias
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Linear(64, 10, bias=False)     # Wrong: missing bias in final layer
)
check_bias_usage(model)
# Output:
# Warning: 0.weight has bias=True before BatchNorm
# Warning: 3.weight has bias=False but no BatchNorm follows

# Impact on model performance:
1. Unnecessary bias: Wastes parameters, slightly slower training
2. Missing output bias: Can harm model's ability to fit the data
"""
```

## 6. Tensor Reshaping Operations

### The Problem
`view()` and `permute()` serve different purposes, but are often confused. `view()` assumes contiguous memory layout, while `permute()` changes dimension order.

### The Solution
Understand when to use each operation:

```python
def tensor_reshaping_guide():
    # Create a sample tensor
    x = torch.randn(2, 3, 4)
    print("Original tensor:")
    debug_tensor_shape(x, "x")
    
    # view() example - reshapes tensor assuming contiguous memory
    x_reshaped = x.view(2, 12)  # Combines last two dimensions
    print("\nAfter view():")
    debug_tensor_shape(x_reshaped, "x_reshaped")
    
    # permute() example - rearranges dimensions
    x_permuted = x.permute(2, 0, 1)  # Changes dimension order
    print("\nAfter permute():")
    debug_tensor_shape(x_permuted, "x_permuted")
    
    # Common mistake (DON'T DO THIS):
    x_permuted = x.permute(1, 0, 2)
    try:
        x_wrong = x_permuted.view(6, 4)
        print("\nThis might fail or give wrong results!")
    except RuntimeError as e:
        print("\nError:", str(e))
    
    # Correct approach:
    x_permuted = x.permute(1, 0, 2)
    x_correct = x_permuted.contiguous().view(6, 4)
    print("\nCorrect approach result:")
    debug_tensor_shape(x_correct, "x_correct")

# Helper function to check tensor operations
def debug_tensor_shape(tensor, name="tensor"):
    print(f"{name}.shape:", tensor.shape)
    print(f"{name}.stride():", tensor.stride())
    print(f"{name}.is_contiguous():", tensor.is_contiguous())

# Example usage and output:
"""
Original tensor:
x.shape: torch.Size([2, 3, 4])
x.stride(): (12, 4, 1)
x.is_contiguous(): True

After view():
x_reshaped.shape: torch.Size([2, 12])
x_reshaped.stride(): (12, 1)
x_reshaped.is_contiguous(): True

After permute():
x_permuted.shape: torch.Size([4, 2, 3])
x_permuted.stride(): (1, 12, 4)
x_permuted.is_contiguous(): False

Error: view size is not compatible with input tensor's size
```

## Best Practices Summary

1. Always verify your model can learn by overfitting a single batch first
2. Create explicit train/eval mode switching functions
3. Make gradient zeroing a consistent habit
4. Double-check loss function input requirements
5. Use bias appropriately with BatchNorm
6. Understand tensor reshaping operations

These practices will help catch common issues early and make your neural network training more robust.

## Contributing

Feel free to submit issues and enhancement requests!

## References

- Original [Tweet](https://x.com/karpathy/status/1013244313327681536) by Andrej Karpathy
- Extended Blog Post: [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) by Andrej Karpathy
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)

Another Deep Learning Related Resources from [Andrej Karpathy](https://github.com/karpathy), The Legend Himself.
- [Neural Networks: Zero to Hero, Andrej Karpathy](https://karpathy.ai/zero-to-hero.html)
- [Hacker's guide to Neural Networks](https://karpathy.github.io/neuralnets/)
- [Deep Learning for Computer Vision (Andrej Karpathy, OpenAI)](https://www.youtube.com/watch?v=u6aEYuemt0M)
- [Stanford CS25: V2 I Introduction to Transformers w/ Andrej Karpathy](https://www.youtube.com/watch?v=XfpMkf4rD6E)

---
[Back-to-Top](#table-of-contents)

