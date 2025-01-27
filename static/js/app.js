// Ensure the DOM is fully loaded before executing scripts
document.addEventListener('DOMContentLoaded', function() {
    // Select the file input and form elements
    const fileInput = document.getElementById('image');
    const uploadForm = document.querySelector('form');

    // Check if the file input and form exist on the page
    if (fileInput && uploadForm) {
        // Add an event listener to the form to handle form submission
        uploadForm.addEventListener('submit', function(event) {
            // Check if a file has been selected
            if (!fileInput.files.length) {
                // Prevent form submission if no file is selected
                event.preventDefault();
                alert('Please select an image file to upload.');
            }
        });
    }

    // Select all category links
    const categoryLinks = document.querySelectorAll('.category-link');

    // Add event listeners to each category link for user feedback
    categoryLinks.forEach(function(link) {
        link.addEventListener('click', function() {
            // Provide visual feedback when a category is selected
            link.classList.add('selected');
        });
    });
});
