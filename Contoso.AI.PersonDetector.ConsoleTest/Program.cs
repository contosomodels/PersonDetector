using Contoso.AI;
using System.Drawing;

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
