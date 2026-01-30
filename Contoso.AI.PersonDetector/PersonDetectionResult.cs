using System.Collections.Generic;

namespace Contoso.AI
{
    /// <summary>
    /// Represents the result of person detection on an image.
    /// </summary>
    public class PersonDetectionResult
    {
        /// <summary>
        /// Gets or sets the list of detected people in the image.
        /// </summary>
        public List<Detection> People { get; set; } = new();

        /// <summary>
        /// Gets or sets the total count of detected people.
        /// </summary>
        public int TotalPeopleCount { get; set; }
    }
}
