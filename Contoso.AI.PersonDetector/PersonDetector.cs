using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Windows.AI.MachineLearning;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace Contoso.AI
{
    /// <summary>
    /// Provides person detection capabilities using YoloX models on NPU hardware acceleration.
    /// </summary>
    public class PersonDetector : IDisposable
    {
        private readonly OrtEnv _env;
        private readonly InferenceSession _session;
        private readonly OrtIoBinding? _ioBinding;
        private readonly string _inputName;

        private const int InputWidth = 640;
        private const int InputHeight = 640;
        private const float ConfidenceThreshold = 0.8f;
        private const float NmsThreshold = 0.45f;

        private PersonDetector(OrtEnv env, InferenceSession session)
        {
            _env = env;
            _session = session;

            if (session != null)
            {
                // Get input metadata
                var inputMeta = _session.InputMetadata.First();
                _inputName = inputMeta.Key;

                // Try to set up IO binding for better performance (optional)
                try
                {
                    _ioBinding = session.CreateIoBinding();
                    Debug.WriteLine("[PersonDetector] IO Binding created successfully for zero-copy operations");
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"[PersonDetector] IO Binding not available: {ex.Message}");
                }
            }
        }

        /// <summary>
        /// Gets the ready state of the person detection feature.
        /// </summary>
        /// <returns>The ready state indicating if the feature can be used.</returns>
        public static AIFeatureReadyState GetReadyState()
        {
            const string modelPath = "Models/Yolo-X_w8a8/model.onnx";

            try
            {
                // Check if model file exists
                if (!File.Exists(modelPath))
                {
                    Debug.WriteLine($"[PersonDetector] Model file not found: {modelPath}");
                    return AIFeatureReadyState.NotReady;
                }

                // Check if Windows ML EP catalog is available
                var catalog = ExecutionProviderCatalog.GetDefault();
                var qnnProvider = catalog.FindAllProviders().FirstOrDefault(i => i.Name == "QNNExecutionProvider");

                if (qnnProvider == null)
                {
                    Debug.WriteLine("[PersonDetector] QNN Execution Provider not found");
                    return AIFeatureReadyState.NotReady;
                }

                if (qnnProvider.ReadyState == ExecutionProviderReadyState.NotPresent)
                {
                    Debug.WriteLine("[PersonDetector] QNN Execution Provider not present, needs download");
                    return AIFeatureReadyState.NotReady;
                }

                return AIFeatureReadyState.Ready;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[PersonDetector] Error checking ready state: {ex.Message}");
                return AIFeatureReadyState.NotReady;
            }
        }

        /// <summary>
        /// Ensures the person detection feature is ready by downloading necessary dependencies.
        /// </summary>
        /// <returns>A task that represents the asynchronous operation. The task result contains the preparation result.</returns>
        public static async Task<AIFeatureReadyResult> EnsureReadyAsync()
        {
            const string modelPath = "Models/Yolo-X_w8a8/model.onnx";

            try
            {
                // Check if model file exists
                if (!File.Exists(modelPath))
                {
                    throw new FileNotFoundException($"Model file not found: {modelPath}");
                }

                // Get the Windows ML EP catalog
                var catalog = ExecutionProviderCatalog.GetDefault();

                // Get the QNN EP provider info
                var qnnProvider = catalog.FindAllProviders().FirstOrDefault(i => i.Name == "QNNExecutionProvider");

                // If its ReadyState is NotPresent, download the EP
                if (qnnProvider != null && qnnProvider.ReadyState == ExecutionProviderReadyState.NotPresent)
                {
                    Debug.WriteLine("[PersonDetector] Downloading QNN Execution Provider...");
                    await qnnProvider.EnsureReadyAsync();
                }

                // Register all EPs with ONNX Runtime
                await catalog.RegisterCertifiedAsync();

                Debug.WriteLine("[PersonDetector] Person detection feature is ready");
                return AIFeatureReadyResult.Success();
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[PersonDetector] Failed to ensure ready: {ex.Message}");
                return AIFeatureReadyResult.Failed(ex);
            }
        }

        /// <summary>
        /// Creates a new person detector instance with NPU acceleration.
        /// </summary>
        /// <returns>A task that represents the asynchronous operation. The task result contains the initialized detector.</returns>
        /// <exception cref="FileNotFoundException">Thrown when the model file is not found.</exception>
        /// <exception cref="InvalidOperationException">Thrown when the QNN Execution Provider is not available.</exception>
        public static async Task<PersonDetector> CreateAsync()
        {
            const string modelPath = "Models/Yolo-X_w8a8/model.onnx";

            if (!File.Exists(modelPath))
            {
                throw new FileNotFoundException($"Model file not found: {modelPath}");
            }

            // Ensure dependencies are ready
            var readyResult = await EnsureReadyAsync();
            if (readyResult.Status != AIFeatureReadyResultState.Success)
            {
                throw readyResult.ExtendedError ?? new InvalidOperationException("Failed to prepare person detection feature");
            }

            // Get the ORT environment
            var env = OrtEnv.Instance();

            // Get the available EP devices
            var epDevices = env.GetEpDevices();

            // Get the QNN NPU EP device
            var ep = epDevices.FirstOrDefault(i => i.EpName == "QNNExecutionProvider" && i.HardwareDevice.Type == OrtHardwareDeviceType.NPU);
            if (ep == null)
            {
                ep = epDevices.First(i => i.EpName == "CPUExecutionProvider");
            }

            // Configure session options to use the QNN EP or CPU EP
            var sessionOptions = new SessionOptions();
            sessionOptions.AppendExecutionProvider(env, new[] { ep }, null);

            // Create the inference session with the specified options
            var session = new InferenceSession(modelPath, sessionOptions);

            Debug.WriteLine($"[PersonDetector] Created with model: {modelPath}");
            return new PersonDetector(env, session);
        }

        /// <summary>
        /// Detects people in the provided bitmap image.
        /// </summary>
        /// <param name="bitmap">The bitmap image to analyze.</param>
        /// <returns>The detection result containing information about detected people.</returns>
        /// <exception cref="ArgumentNullException">Thrown when bitmap is null.</exception>
        public PersonDetectionResult DetectPeople(Bitmap bitmap)
        {
            if (bitmap == null)
                throw new ArgumentNullException(nameof(bitmap));

            if (_session == null)
            {
                return new PersonDetectionResult();
            }

            Debug.WriteLine($"[PersonDetector] ===== DETECTION START =====");
            Debug.WriteLine($"[PersonDetector] DetectPeople called. Bitmap size: {bitmap.Width}x{bitmap.Height}");

            var inputTensor = PreprocessImageFast(bitmap);
            var inputValue = NamedOnnxValue.CreateFromTensor(_inputName, inputTensor);

            using var results = _session.Run(new[] { inputValue });
            Debug.WriteLine($"[PersonDetector] Outputs returned: {results.Count}");

            // Log all output names and shapes
            foreach (var output in results)
            {
                var dims = output.Value is DenseTensor<byte> bt ? bt.Dimensions.ToArray() :
                          output.Value is DenseTensor<float> ft ? ft.Dimensions.ToArray() : Array.Empty<int>();
                Debug.WriteLine($"[PersonDetector] Output '{output.Name}': {string.Join('x', dims)}");
            }

            var detections = DecodeYoloXOutput(results, bitmap.Width, bitmap.Height);

            Debug.WriteLine($"[PersonDetector] Raw detections count: {detections.Count}");

            var filtered = detections.Where(d => d.Confidence >= ConfidenceThreshold)
                                    .OrderByDescending(d => d.Confidence)
                                    .ToList();

            Debug.WriteLine($"[PersonDetector] After confidence filter ({ConfidenceThreshold}): {filtered.Count}");

            var nms = ApplyNMS(filtered, NmsThreshold);

            Debug.WriteLine($"[PersonDetector] After NMS: {nms.Count}");

            foreach (var d in nms)
            {
                Debug.WriteLine($"[PersonDetector] Person detected -> Conf={d.Confidence:F3} " +
                    $"Box=({d.BoundingBox.X:F0},{d.BoundingBox.Y:F0},{d.BoundingBox.Width:F0},{d.BoundingBox.Height:F0})");
            }

            Debug.WriteLine($"[PersonDetector] ===== DETECTION END =====");

            return new PersonDetectionResult
            {
                People = nms,
                TotalPeopleCount = nms.Count
            };
        }

        /// <summary>
        /// Fast image preprocessing using unsafe code and LockBits for 10-100x speedup.
        /// </summary>
        private unsafe DenseTensor<byte> PreprocessImageFast(Bitmap bitmap)
        {
            // Resize to model input size
            using var resized = new Bitmap(bitmap, new Size(InputWidth, InputHeight));

            // YoloX expects NCHW format: [batch, channels, height, width]
            var tensor = new DenseTensor<byte>(new[] { 1, 3, InputHeight, InputWidth });

            // Lock bitmap for fast pixel access
            var bitmapData = resized.LockBits(
                new Rectangle(0, 0, InputWidth, InputHeight),
                ImageLockMode.ReadOnly,
                PixelFormat.Format24bppRgb);

            try
            {
                byte* srcPtr = (byte*)bitmapData.Scan0;
                int stride = bitmapData.Stride;

                // Get tensor buffer
                var tensorSpan = tensor.Buffer.Span;

                // Calculate channel offsets in the tensor
                int channelSize = InputHeight * InputWidth;
                int rOffset = 0;
                int gOffset = channelSize;
                int bOffset = 2 * channelSize;

                for (int y = 0; y < InputHeight; y++)
                {
                    byte* row = srcPtr + (y * stride);
                    int rowOffset = y * InputWidth;

                    for (int x = 0; x < InputWidth; x++)
                    {
                        int idx = rowOffset + x;

                        // BGR format in bitmap
                        byte b = row[x * 3 + 0];
                        byte g = row[x * 3 + 1];
                        byte r = row[x * 3 + 2];

                        // Store in NCHW format
                        tensorSpan[rOffset + idx] = r;
                        tensorSpan[gOffset + idx] = g;
                        tensorSpan[bOffset + idx] = b;
                    }
                }
            }
            finally
            {
                resized.UnlockBits(bitmapData);
            }

            return tensor;
        }

        private List<Detection> DecodeYoloXOutput(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results,
                                                   int origWidth,
                                                   int origHeight)
        {
            var detections = new List<Detection>();

            // YoloX models can have separate outputs for boxes, scores, and classes
            DenseTensor<byte>? boxesTensor = null;
            DenseTensor<byte>? scoresTensor = null;
            DenseTensor<byte>? classesTensor = null;

            DenseTensor<float>? boxesTensorFloat = null;
            DenseTensor<float>? scoresTensorFloat = null;
            DenseTensor<float>? classesTensorFloat = null;

            foreach (var output in results)
            {
                var name = output.Name.ToLowerInvariant();

                if (output.Value is DenseTensor<byte> bt)
                {
                    if (name.Contains("box"))
                        boxesTensor = bt;
                    else if (name.Contains("score") || name.Contains("conf"))
                        scoresTensor = bt;
                    else if (name.Contains("class") || name.Contains("label"))
                        classesTensor = bt;
                    else if (boxesTensor == null)
                        boxesTensor = bt;
                    else if (scoresTensor == null)
                        scoresTensor = bt;
                    else if (classesTensor == null)
                        classesTensor = bt;
                }
                else if (output.Value is DenseTensor<float> ft)
                {
                    if (name.Contains("box"))
                        boxesTensorFloat = ft;
                    else if (name.Contains("score") || name.Contains("conf"))
                        scoresTensorFloat = ft;
                    else if (name.Contains("class") || name.Contains("label"))
                        classesTensorFloat = ft;
                    else if (boxesTensorFloat == null)
                        boxesTensorFloat = ft;
                    else if (scoresTensorFloat == null)
                        scoresTensorFloat = ft;
                    else if (classesTensorFloat == null)
                        classesTensorFloat = ft;
                }
            }

            // Decode based on what we found
            if (boxesTensor != null)
            {
                detections.AddRange(DecodeMultiOutputByte(boxesTensor, scoresTensor, classesTensor, origWidth, origHeight));
            }
            else if (boxesTensorFloat != null)
            {
                detections.AddRange(DecodeMultiOutputFloat(boxesTensorFloat, scoresTensorFloat, classesTensorFloat, origWidth, origHeight));
            }

            return detections;
        }

        private List<Detection> DecodeMultiOutputByte(DenseTensor<byte> boxes,
                                                       DenseTensor<byte>? scores,
                                                       DenseTensor<byte>? classes,
                                                       int origWidth,
                                                       int origHeight)
        {
            var detections = new List<Detection>();
            var boxDims = boxes.Dimensions.ToArray();

            Debug.WriteLine($"[PersonDetector] Multi-output byte format: boxes={string.Join('x', boxDims)}");
            Debug.WriteLine($"[PersonDetector] Original image dimensions: {origWidth}x{origHeight}");

            if (boxDims.Length != 3 || boxDims[2] != 4)
            {
                Debug.WriteLine($"[PersonDetector] Unexpected box dimensions, expected [1, N, 4]");
                return detections;
            }

            int numBoxes = boxDims[1];

            // Sample boxes to understand the value distribution and quantization scheme
            var sampleValues = new List<byte>();
            for (int i = 0; i < Math.Min(100, numBoxes); i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    sampleValues.Add(boxes[0, i, j]);
                }
            }

            byte minVal = 255, maxVal = 0;
            if (sampleValues.Count > 0)
            {
                minVal = sampleValues.Min();
                maxVal = sampleValues.Max();
                double avgVal = sampleValues.Select(b => (int)b).Average();

                Debug.WriteLine($"[PersonDetector] Box value distribution: min={minVal} max={maxVal} avg={avgVal:F1}");
            }

            // Infer quantization parameters from the data
            float scale = (float)InputWidth / (maxVal - minVal);
            float zeroPoint = minVal;

            Debug.WriteLine($"[PersonDetector] Inferred quantization: scale={scale:F4} zero_point={zeroPoint:F1}");
            Debug.WriteLine($"[PersonDetector] This maps byte range [{minVal}, {maxVal}] to coord range [0, {InputWidth}]");

            // Process all boxes
            for (int i = 0; i < numBoxes; i++)
            {
                // Dequantize bounding box coordinates
                float b0 = (boxes[0, i, 0] - zeroPoint) * scale;
                float b1 = (boxes[0, i, 1] - zeroPoint) * scale;
                float b2 = (boxes[0, i, 2] - zeroPoint) * scale;
                float b3 = (boxes[0, i, 3] - zeroPoint) * scale;

                // Clamp to valid range
                b0 = Math.Max(0, Math.Min(InputWidth, b0));
                b1 = Math.Max(0, Math.Min(InputHeight, b1));
                b2 = Math.Max(0, Math.Min(InputWidth, b2));
                b3 = Math.Max(0, Math.Min(InputHeight, b3));

                // Get confidence score
                float confidence = 1.0f;
                if (scores != null)
                {
                    var scoreDims = scores.Dimensions.ToArray();
                    if (scoreDims.Length == 2 && scoreDims[1] > i)
                    {
                        confidence = scores[0, i] / 255.0f;
                    }
                    else if (scoreDims.Length == 3 && scoreDims[1] > i && scoreDims[2] > 0)
                    {
                        confidence = scores[0, i, 0] / 255.0f;
                    }
                }

                // Early filtering
                if (confidence < 0.05f) continue;

                // Convert from model coordinate space to original image space
                float scaleX = (float)origWidth / InputWidth;
                float scaleY = (float)origHeight / InputHeight;

                float x1 = b0 * scaleX;
                float y1 = b1 * scaleY;
                float x2 = b2 * scaleX;
                float y2 = b3 * scaleY;

                // Convert to x, y, width, height format
                float x = x1;
                float y = y1;
                float w = x2 - x1;
                float h = y2 - y1;

                // Skip very small or invalid boxes
                if (w < 4 || h < 4 || x < 0 || y < 0 || w > origWidth * 2 || h > origHeight * 2)
                    continue;

                detections.Add(new Detection
                {
                    ClassName = "Person",
                    Confidence = confidence,
                    BoundingBox = new RectangleF(x, y, w, h)
                });
            }

            Debug.WriteLine($"[PersonDetector] Decoded {detections.Count} detections from multi-output byte format");
            return detections;
        }

        private List<Detection> DecodeMultiOutputFloat(DenseTensor<float> boxes,
                                                        DenseTensor<float>? scores,
                                                        DenseTensor<float>? classes,
                                                        int origWidth,
                                                        int origHeight)
        {
            var detections = new List<Detection>();
            var boxDims = boxes.Dimensions.ToArray();

            Debug.WriteLine($"[PersonDetector] Multi-output float format: boxes={string.Join('x', boxDims)}");

            if (boxDims.Length != 3 || boxDims[2] != 4)
            {
                Debug.WriteLine($"[PersonDetector] Unexpected box dimensions, expected [1, N, 4]");
                return detections;
            }

            int numBoxes = boxDims[1];

            for (int i = 0; i < numBoxes; i++)
            {
                float b0 = boxes[0, i, 0];
                float b1 = boxes[0, i, 1];
                float b2 = boxes[0, i, 2];
                float b3 = boxes[0, i, 3];

                float confidence = 1.0f;
                if (scores != null)
                {
                    var scoreDims = scores.Dimensions.ToArray();
                    if (scoreDims.Length == 2 && scoreDims[1] > i)
                    {
                        confidence = scores[0, i];
                    }
                    else if (scoreDims.Length == 3 && scoreDims[1] > i && scoreDims[2] > 0)
                    {
                        confidence = scores[0, i, 0];
                    }
                }

                if (confidence < 0.05f) continue;

                // Scale coordinates from model space to original image space
                float scaleX = (float)origWidth / InputWidth;
                float scaleY = (float)origHeight / InputHeight;

                float x1, y1, x2, y2;

                if (b0 <= 1.0f && b1 <= 1.0f && b2 <= 1.0f && b3 <= 1.0f)
                {
                    // Normalized coordinates
                    x1 = b0 * InputWidth * scaleX;
                    y1 = b1 * InputHeight * scaleY;
                    x2 = b2 * InputWidth * scaleX;
                    y2 = b3 * InputHeight * scaleY;
                }
                else
                {
                    // Already in pixel space
                    x1 = b0 * scaleX;
                    y1 = b1 * scaleY;
                    x2 = b2 * scaleX;
                    y2 = b3 * scaleY;
                }

                // Convert to x, y, width, height format
                float x = x1;
                float y = y1;
                float w = x2 - x1;
                float h = y2 - y1;

                if (w < 4 || h < 4 || x < 0 || y < 0 || w > origWidth * 2 || h > origHeight * 2)
                    continue;

                detections.Add(new Detection
                {
                    ClassName = "Person",
                    Confidence = confidence,
                    BoundingBox = new RectangleF(x, y, w, h)
                });
            }

            Debug.WriteLine($"[PersonDetector] Decoded {detections.Count} detections from multi-output float format");
            return detections;
        }

        private List<Detection> ApplyNMS(List<Detection> detections, float nmsThreshold)
        {
            if (detections == null || detections.Count == 0)
                return new List<Detection>();

            var result = new List<Detection>();
            var sorted = detections.OrderByDescending(d => d.Confidence).ToList();

            Debug.WriteLine($"[PersonDetector][NMS] Starting NMS on {sorted.Count} detections, threshold={nmsThreshold}");

            while (sorted.Count > 0)
            {
                var current = sorted[0];
                result.Add(current);
                sorted.RemoveAt(0);

                sorted.RemoveAll(det => CalculateIoU(current.BoundingBox, det.BoundingBox) > nmsThreshold);
            }

            Debug.WriteLine($"[PersonDetector][NMS] NMS result count={result.Count}");
            return result;
        }

        private float CalculateIoU(RectangleF box1, RectangleF box2)
        {
            var intersection = RectangleF.Intersect(box1, box2);
            if (intersection.IsEmpty) return 0;

            float interArea = intersection.Width * intersection.Height;
            float unionArea = box1.Width * box1.Height + box2.Width * box2.Height - interArea;

            return interArea / unionArea;
        }

        /// <summary>
        /// Releases all resources used by the PersonDetector.
        /// </summary>
        public void Dispose()
        {
            _ioBinding?.Dispose();
            _session?.Dispose();
        }
    }
}
