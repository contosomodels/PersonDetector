using System.Drawing;

namespace Contoso.AI
{
    /// <summary>
    /// Represents a detected person or object with bounding box and confidence.
    /// </summary>
    public class Detection
    {
        /// <summary>
        /// Gets or sets the class name of the detected object (e.g., "Person").
        /// </summary>
        public string ClassName { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the confidence score of the detection (0.0 to 1.0).
        /// </summary>
        public float Confidence { get; set; }

        /// <summary>
        /// Gets or sets the bounding box of the detected object.
        /// </summary>
        public RectangleF BoundingBox { get; set; }
    }
}
