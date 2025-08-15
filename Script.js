function showModal(title, message) {
    document.getElementById('modal-title').innerText = title;
    document.getElementById('modal-message').innerText = message;
    document.getElementById('custom-modal').classList.remove('hidden');
}

// Function to close custom modal
function closeModal() {
    document.getElementById('custom-modal').classList.add('hidden');
}

// Attach event listener to modal OK button
document.getElementById('modal-ok-button').addEventListener('click', closeModal);

// --- Login Page Logic ---
document.getElementById('login-form').addEventListener('submit', function(e) {
    e.preventDefault();
    // Simulate successful login for any input
    // Open the dashboard in a new tab
    window.open('dashboard.html', '_blank');
});