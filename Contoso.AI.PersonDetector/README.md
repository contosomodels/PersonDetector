# Contoso.AI.PersonDetector

AI-powered person detection for images using ONNX Runtime and the Yolo-X model.

> **Windows Only**: This package requires Windows 10 SDK version 19041 or later and uses Windows-specific AI APIs.

## Features

- Detects persons in images with bounding boxes and confidence scores
- Performance and Efficiency modes for different hardware configurations
- Async API for non-blocking operations
- Automatic model download at build time

## Installation

```bash
dotnet add package Contoso.AI.PersonDetector
```

**Platform Requirements:**
- Windows 10 version 2004 (build 19041) or later
- Not compatible with: Linux, macOS, or older Windows versions

## Model Download

**Important**: This package uses a Yolo-X ONNX model that is **automatically downloaded at build time**.

- The model is downloaded to `obj/Models/Yolo-X_w8a8.onnx` (not tracked by git)
- Download happens only once (cached for subsequent builds)
- The model file is automatically copied to your output directory (`bin/.../Models/`)
- **No need to add to `.gitignore`** - the `obj/` folder is already ignored by default

## Usage

```csharp
using Contoso.AI.PersonDetector;

// Initialize (call once at startup)
var readyResult = await PersonDetector.EnsureReadyAsync();
if (readyResult.Status != AIFeatureReadyResultState.Success)
{
    // Handle initialization failure
    return;
}

// Create detector instance
var detector = await PersonDetector.CreateAsync(PerformanceMode.Performance);

// Detect persons in an image
var result = await detector.DetectAsync(imageData);

Console.WriteLine($"Persons detected: {result.Detections.Count}");
foreach (var detection in result.Detections)
{
    Console.WriteLine($"  Confidence: {detection.Confidence:P}, Box: {detection.BoundingBox}");
}
Console.WriteLine($"Inference Time: {result.InferenceTimeMs}ms");

// Don't forget to dispose
detector.Dispose();
```

## Performance Modes

- `PerformanceMode.Performance` - Optimized for speed (uses GPU/NPU if available)
- `PerformanceMode.Efficiency` - Optimized for power efficiency (uses CPU)

## Model Information

- **Source**: [Qualcomm/Yolo-X on HuggingFace](https://huggingface.co/qualcomm/Yolo-X)
- **Type**: Yolo-X ONNX model (quantized w8a8)
- **License**: Check the model repository for license information

## Requirements

- .NET 8.0 or later
- Windows 10 SDK 19041 or later
- Internet connection for initial model download

## License

MIT

