using Contoso.AI;
using System.Drawing;
using System.Drawing.Drawing2D;

Console.WriteLine("=== Contoso.AI.PersonDetector Console Test ===");
Console.WriteLine();

// Path to sample image
const string imagePath = "Assets/SampleImage.png";

if (!File.Exists(imagePath))
{
    Console.WriteLine($"ERROR: Sample image not found at: {imagePath}");
    return;
}

Console.WriteLine($"Loading image: {imagePath}");
Console.WriteLine();

try
{
    // Check if the person detection feature is ready
    Console.WriteLine("Checking if PersonDetector is ready...");
    var readyState = PersonDetector.GetReadyState();
    
    if (readyState != AIFeatureReadyState.Ready)
    {
        Console.WriteLine($"PersonDetector is not ready. State: {readyState}");
        Console.WriteLine("Attempting to prepare the feature...");
        
        var readyResult = await PersonDetector.EnsureReadyAsync();
        
        if (readyResult.Status != AIFeatureReadyResultState.Success)
        {
            Console.WriteLine($"ERROR: Failed to prepare PersonDetector: {readyResult.ExtendedError?.Message}");
            return;
        }
        
        Console.WriteLine("PersonDetector is now ready!");
    }
    else
    {
        Console.WriteLine("PersonDetector is ready!");
    }
    
    Console.WriteLine();

    // Create the detector
    Console.WriteLine("Creating PersonDetector instance...");
    using var detector = await PersonDetector.CreateAsync();
    Console.WriteLine("PersonDetector created successfully!");
    Console.WriteLine();

    // Load the image
    Console.WriteLine("Loading and analyzing image...");
    using var bitmap = new Bitmap(imagePath);
    Console.WriteLine($"Image size: {bitmap.Width}x{bitmap.Height}");
    Console.WriteLine();

    // Detect people
    var result = detector.DetectPeople(bitmap);

    // Output results
    Console.WriteLine("=== DETECTION RESULTS ===");
    Console.WriteLine($"Total people detected: {result.TotalPeopleCount}");
    Console.WriteLine();

    if (result.TotalPeopleCount > 0)
    {
        Console.WriteLine("Detected people:");
        for (int i = 0; i < result.People.Count; i++)
        {
            var person = result.People[i];
            Console.WriteLine($"  Person #{i + 1}:");
            Console.WriteLine($"    Confidence: {person.Confidence:P2} ({person.Confidence:F4})");
            Console.WriteLine($"    Bounding Box:");
            Console.WriteLine($"      X: {person.BoundingBox.X:F1}");
            Console.WriteLine($"      Y: {person.BoundingBox.Y:F1}");
            Console.WriteLine($"      Width: {person.BoundingBox.Width:F1}");
            Console.WriteLine($"      Height: {person.BoundingBox.Height:F1}");
            Console.WriteLine();
        }

        // Create annotated image with bounding boxes
        Console.WriteLine("Creating annotated image with bounding boxes...");
        using var annotatedImage = DrawBoundingBoxes(bitmap, result.People);
        
        // Copy to clipboard
        Console.WriteLine("Copying annotated image to clipboard...");
        CopyToClipboard(annotatedImage);
        Console.WriteLine("✓ Annotated image copied to clipboard!");
        Console.WriteLine();
    }
    else
    {
        Console.WriteLine("No people detected in the image.");
    }

    Console.WriteLine("=== TEST COMPLETE ===");
}
catch (Exception ex)
{
    Console.WriteLine($"ERROR: {ex.GetType().Name}: {ex.Message}");
    Console.WriteLine($"Stack trace: {ex.StackTrace}");
}

static Bitmap DrawBoundingBoxes(Bitmap originalImage, List<Detection> detections)
{
    // Create a copy of the image to draw on
    var annotated = new Bitmap(originalImage);
    
    using var graphics = Graphics.FromImage(annotated);
    graphics.SmoothingMode = SmoothingMode.AntiAlias;
    
    // Set up drawing tools
    using var pen = new Pen(Color.LimeGreen, 3);
    using var brush = new SolidBrush(Color.FromArgb(180, 0, 255, 0)); // Semi-transparent green
    using var font = new Font("Arial", 12, FontStyle.Bold);
    using var textBrush = new SolidBrush(Color.White);
    using var textBackground = new SolidBrush(Color.FromArgb(200, 0, 200, 0));
    
    // Draw each detection
    for (int i = 0; i < detections.Count; i++)
    {
        var detection = detections[i];
        var box = detection.BoundingBox;
        
        // Draw bounding box
        graphics.DrawRectangle(pen, box.X, box.Y, box.Width, box.Height);
        
        // Draw label with confidence
        string label = $"Person {i + 1}: {detection.Confidence:P0}";
        var labelSize = graphics.MeasureString(label, font);
        
        // Draw label background
        var labelRect = new RectangleF(
            box.X,
            box.Y - labelSize.Height - 4,
            labelSize.Width + 8,
            labelSize.Height + 4);
        
        // Make sure label is within image bounds
        if (labelRect.Y < 0)
        {
            labelRect.Y = box.Y + 2;
        }
        
        graphics.FillRectangle(textBackground, labelRect);
        graphics.DrawString(label, font, textBrush, labelRect.X + 4, labelRect.Y + 2);
    }
    
    return annotated;
}

static void CopyToClipboard(Bitmap image)
{
    // Windows clipboard requires STA thread
    var thread = new Thread(() =>
    {
        System.Windows.Forms.Clipboard.SetImage(image);
    });
    
    thread.SetApartmentState(ApartmentState.STA);
    thread.Start();
    thread.Join();
}
