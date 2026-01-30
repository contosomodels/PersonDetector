# Contoso.AI.PersonDetector

AI-powered person detection for images using ONNX Runtime and the Yolo-X model.

> **Windows Only**: This package requires Windows 10 SDK version 19041 or later and uses Windows-specific AI APIs.

## Features

- Detects persons in images with bounding boxes and confidence scores
- NPU acceleration using QNN Execution Provider (falls back to CPU on other devices)
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

- The model is downloaded to `obj/Models/Yolo-X_w8a8/` (not tracked by git)
- Download happens only once (cached for subsequent builds)
- The model files (`model.onnx` and `model.data`) are automatically copied to your output directory (`bin/.../Models/Yolo-X_w8a8/`)
- **No need to add to `.gitignore`** - the `obj/` folder is already ignored by default

## Usage

```csharp
using Contoso.AI;
using System.Drawing;

// Check if the person detection feature is ready
var readyState = PersonDetector.GetReadyState();

if (readyState != AIFeatureReadyState.Ready)
{
    // Prepare the feature (downloads QNN Execution Provider if needed)
    var readyResult = await PersonDetector.EnsureReadyAsync();
    if (readyResult.Status != AIFeatureReadyResultState.Success)
    {
        // Handle initialization failure
        Console.WriteLine($"Failed to initialize: {readyResult.ExtendedError?.Message}");
        return;
    }
}

// Create detector instance
using var detector = await PersonDetector.CreateAsync();

// Load and detect people in an image
using var bitmap = new Bitmap("path/to/image.png");
var result = detector.DetectPeople(bitmap);

Console.WriteLine($"Total people detected: {result.TotalPeopleCount}");
foreach (var person in result.People)
{
    Console.WriteLine($"  Confidence: {person.Confidence:P2}");
    Console.WriteLine($"  Bounding Box: X={person.BoundingBox.X:F1}, Y={person.BoundingBox.Y:F1}, " +
                      $"Width={person.BoundingBox.Width:F1}, Height={person.BoundingBox.Height:F1}");
}

// Don't forget to dispose
detector.Dispose();
```

## API Reference

### PersonDetector Class

**Static Methods:**
- `AIFeatureReadyState GetReadyState()` - Checks if the person detection feature is ready
- `Task<AIFeatureReadyResult> EnsureReadyAsync()` - Ensures dependencies are downloaded and ready
- `Task<PersonDetector> CreateAsync()` - Creates a new detector instance with NPU acceleration

**Instance Methods:**
- `PersonDetectionResult DetectPeople(Bitmap bitmap)` - Detects people in the provided bitmap image

### PersonDetectionResult Class

**Properties:**
- `List<Detection> People` - List of detected people
- `int TotalPeopleCount` - Total count of detected people

### Detection Class

**Properties:**
- `string ClassName` - Class name (e.g., "Person")
- `float Confidence` - Confidence score (0.0 to 1.0)
- `RectangleF BoundingBox` - Bounding box with X, Y, Width, Height

## Performance

The PersonDetector uses NPU (Neural Processing Unit) acceleration via the QNN Execution Provider for optimal performance on compatible hardware. The QNN EP is automatically downloaded on first use if not already present.

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

