/* ==========================================================================
   Dashboard Interactions & API Integrations — script.js
   ========================================================================== */

// Global Variables
let currentTab = 'draw';
let selectedFile = null;

// Drawing Canvas Variables
const canvas = document.getElementById('digit-canvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Initialize Elements
document.addEventListener('DOMContentLoaded', () => {
    // Set up drawing canvas properties
    initCanvas();
    
    // Setup File Upload listeners
    initFileUpload();
    
    // Setup Button Click Actions
    initActions();
});

/* ==========================================================================
   Tab Navigation Section
   ========================================================================== */
function switchTab(tab) {
    if (tab === currentTab) return;
    
    currentTab = tab;
    
    // Reset view buttons active state
    document.getElementById('tab-draw-btn').classList.toggle('active', tab === 'draw');
    document.getElementById('tab-upload-btn').classList.toggle('active', tab === 'upload');
    
    // Toggle content displays
    document.getElementById('tab-draw-content').classList.toggle('active', tab === 'draw');
    document.getElementById('tab-upload-content').classList.toggle('active', tab === 'upload');
    
    // Reset output panels to placeholder
    resetOutput();
}

/* ==========================================================================
   Interactive Drawing Canvas Mechanics
   ========================================================================== */
function initCanvas() {
    // Fill canvas background with black (MNIST baseline)
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Drawing style config: thick white strokes
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 18;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    // Mouse Event Listeners
    canvas.addEventListener('mousedown', (e) => {
        isDrawing = true;
        [lastX, lastY] = getCoordinates(e);
    });
    
    canvas.addEventListener('mousemove', draw);
    
    canvas.addEventListener('mouseup', () => isDrawing = false);
    canvas.addEventListener('mouseout', () => isDrawing = false);
    
    // Touch Event Listeners (Mobile Responsive)
    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        isDrawing = true;
        const touch = e.touches[0];
        [lastX, lastY] = getCoordinates(touch);
    });
    
    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        draw(touch);
    });
    
    canvas.addEventListener('touchend', () => isDrawing = false);
}

function getCoordinates(event) {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    return [x, y];
}

function draw(e) {
    if (!isDrawing) return;
    
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    const [x, y] = getCoordinates(e);
    ctx.lineTo(x, y);
    ctx.stroke();
    [lastX, lastY] = [x, y];
}

function clearCanvas() {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    resetOutput();
}

/* ==========================================================================
   File Drop-Zone & Upload Operations
   ========================================================================== */
function initFileUpload() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    
    // Browse Files trigger
    dropZone.addEventListener('click', (e) => {
        // Prevent trigger if clicking on Browse button wrapper label itself
        if (e.target.tagName !== 'LABEL' && e.target.parentElement.tagName !== 'LABEL') {
            fileInput.click();
        }
    });
    
    // File input selection change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
    
    // Drag & Drop handlers
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        }, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        }, false);
    });
    
    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
}

function handleFileSelect(file) {
    // 1. Validation checks
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg'];
    if (!allowedTypes.includes(file.type)) {
        showError("Invalid file type. Please upload a PNG, JPG, or JPEG image.");
        return;
    }
    
    const maxBytes = 2 * 1024 * 1024; // 2MB
    if (file.size > maxBytes) {
        showError("File size exceeds 2MB limit. Please choose a smaller file.");
        return;
    }
    
    selectedFile = file;
    
    // Render file info details UI
    document.getElementById('file-name').textContent = file.name;
    document.getElementById('file-size').textContent = (file.size / 1024).toFixed(1) + " KB";
    
    document.getElementById('drop-zone').style.display = 'none';
    document.getElementById('file-details').style.display = 'flex';
    
    resetOutput();
}

function removeFile() {
    selectedFile = null;
    document.getElementById('file-input').value = '';
    document.getElementById('drop-zone').style.display = 'flex';
    document.getElementById('file-details').style.display = 'none';
    resetOutput();
}

/* ==========================================================================
   Button Triggers & Actions Setup
   ========================================================================== */
function initActions() {
    // Clear canvas trigger
    document.getElementById('clear-canvas-btn').addEventListener('click', clearCanvas);
    
    // Remove upload file trigger
    document.getElementById('remove-file-btn').addEventListener('click', removeFile);
    
    // Prediction triggers
    document.getElementById('predict-draw-btn').addEventListener('click', () => {
        // Convert canvas image to Blob object and submit
        canvas.toBlob((blob) => {
            submitPrediction(blob, 'drawing.png');
        }, 'image/png');
    });
    
    document.getElementById('predict-file-btn').addEventListener('click', () => {
        if (!selectedFile) return;
        submitPrediction(selectedFile, selectedFile.name);
    });
}

/* ==========================================================================
   API Submissions & Output Renderers
   ========================================================================== */
function submitPrediction(fileBlob, filename) {
    // Verify file blob is valid
    if (!fileBlob) {
        showError("No input data available. Please draw or select a file.");
        return;
    }
    
    // Toggle Loading states
    showLoading();
    
    const formData = new FormData();
    formData.append('file', fileBlob, filename);
    
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            // Read error response if json
            return response.json().then(errJson => {
                throw new Error(errJson.error || `HTTP error! Status: ${response.status}`);
            }).catch(() => {
                throw new Error(`Server returned status code: ${response.status}`);
            });
        }
        return response.json();
    })
    .then(data => {
        displayResults(data);
    })
    .catch(error => {
        showError(error.message);
    });
}

function displayResults(data) {
    // Hide loading views
    document.getElementById('result-loader').style.display = 'none';
    document.getElementById('error-display').style.display = 'none';
    
    // Show Output elements
    const outputContainer = document.getElementById('result-output');
    outputContainer.style.display = 'block';
    
    // Populate winning metrics
    document.getElementById('predicted-digit-val').textContent = data.digit;
    document.getElementById('predicted-conf-val').textContent = data.confidence.toFixed(2) + "%";
    document.getElementById('progress-bar-fill').style.width = data.confidence + "%";
    
    // Populate Softmax bars grid
    const barsGrid = document.getElementById('prob-bars-grid');
    barsGrid.innerHTML = ''; // reset previous bars
    
    data.probabilities.forEach((prob, digit) => {
        const isWinning = digit === data.digit;
        const rowClass = isWinning ? 'prob-row winning-digit' : 'prob-row';
        
        const probRow = `
            <div class="${rowClass}">
                <span class="prob-label">${digit}</span>
                <div class="prob-track">
                    <div class="prob-fill" style="width: ${prob}%"></div>
                </div>
                <span class="prob-val">${prob.toFixed(1)}%</span>
            </div>
        `;
        barsGrid.insertAdjacentHTML('beforeend', probRow);
    });
}

function showLoading() {
    document.getElementById('result-placeholder').style.display = 'none';
    document.getElementById('result-output').style.display = 'none';
    document.getElementById('error-display').style.display = 'none';
    document.getElementById('result-loader').style.display = 'flex';
}

function showError(message) {
    document.getElementById('result-placeholder').style.display = 'none';
    document.getElementById('result-output').style.display = 'none';
    document.getElementById('result-loader').style.display = 'none';
    
    const errorContainer = document.getElementById('error-display');
    document.getElementById('error-message').textContent = message;
    errorContainer.style.display = 'flex';
}

function resetOutput() {
    document.getElementById('result-output').style.display = 'none';
    document.getElementById('result-loader').style.display = 'none';
    document.getElementById('error-display').style.display = 'none';
    document.getElementById('result-placeholder').style.display = 'flex';
}
