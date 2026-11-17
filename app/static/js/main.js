/**
 * LUMINE - Main JavaScript
 * Common functionality across all pages.
 */

// ========================================
// Utility Functions
// ========================================

function debounce(fn, delay) {
    let timer;
    return function(...args) {
        clearTimeout(timer);
        timer = setTimeout(() => fn.apply(this, args), delay);
    };
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

function formatTime(dateString) {
    const date = new Date(dateString);
    return date.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit'
    });
}

function getStatusColor(status) {
    const colors = {
        'normal': '#22c55e',
        'attention': '#f59e0b',
        'followup_recommended': '#ef4444',
        'invalid': '#6b7280',
        'pending': '#8b5cf6'
    };
    return colors[status] || '#6b7280';
}

function getStatusLabel(status) {
    const labels = {
        'normal': 'Normal',
        'attention': 'Attention',
        'followup_recommended': 'Follow-up Recommended',
        'invalid': 'Invalid',
        'pending': 'Pending'
    };
    return labels[status] || status;
}

// ========================================
// Toast Notifications
// ========================================

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    
    toast.style.cssText = `
        position: fixed;
        bottom: 24px;
        right: 24px;
        padding: 12px 24px;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        z-index: 9999;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        animation: slideIn 0.3s ease;
        max-width: 400px;
    `;
    
    const colors = {
        'info': '#3b82f6',
        'success': '#22c55e',
        'warning': '#f59e0b',
        'error': '#ef4444'
    };
    
    toast.style.background = colors[type] || colors.info;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Add CSS animations
const styleSheet = document.createElement('style');
styleSheet.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(styleSheet);

// ========================================
// API Helpers
// ========================================

async function apiCall(endpoint, options = {}) {
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
        },
    };
    
    const mergedOptions = {
        ...defaultOptions,
        ...options,
        headers: {
            ...defaultOptions.headers,
            ...(options.headers || {}),
        },
    };
    
    try {
        const response = await fetch(endpoint, mergedOptions);
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || `HTTP ${response.status}`);
        }
        
        return data;
    } catch (error) {
        console.error('API call failed:', error);
        throw error;
    }
}

// ========================================
// Form Validation
// ========================================

function validateEmail(email) {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

function validatePhone(phone) {
    return /^[\d\s\-+()]{10,15}$/.test(phone);
}

function formatInput(input) {
    // Remove extra whitespace
    return input.trim();
}

// ========================================
// Chart/Graph Helpers (for history page)
// ========================================

function createTrendChart(canvasId, data, labels, options = {}) {
    // Simple canvas-based trend chart
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const padding = 40;
    
    // Clear
    ctx.clearRect(0, 0, width, height);
    
    if (!data || data.length < 2) {
        ctx.fillStyle = '#8a9baa';
        ctx.font = '14px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Insufficient data for trend', width/2, height/2);
        return;
    }
    
    // Find min/max
    const min = Math.min(...data) * 0.8;
    const max = Math.max(...data) * 1.2;
    const range = max - min || 1;
    
    // Draw axes
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();
    
    // Draw data points
    const points = data.map((value, i) => {
        const x = padding + (i / (data.length - 1)) * (width - 2 * padding);
        const y = height - padding - ((value - min) / range) * (height - 2 * padding);
        return { x, y, value };
    });
    
    // Draw line
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.beginPath();
    points.forEach((p, i) => {
        if (i === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
    });
    ctx.stroke();
    
    // Draw points
    points.forEach((p) => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
        ctx.fillStyle = '#3b82f6';
        ctx.fill();
    });
    
    // Draw labels
    ctx.fillStyle = '#8a9baa';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'center';
    
    const step = Math.max(1, Math.floor(data.length / 6));
    points.forEach((p, i) => {
        if (i % step === 0 || i === data.length - 1) {
            const label = labels[i] || '';
            ctx.fillText(label, p.x, height - padding + 16);
        }
    });
}

// ========================================
// Initialization
// ========================================

document.addEventListener('DOMContentLoaded', function() {
    // Set time of day greeting
    const greetingEl = document.querySelector('.page-title');
    if (greetingEl) {
        const hour = new Date().getHours();
        let greeting = 'Good morning';
        if (hour >= 12 && hour < 18) greeting = 'Good afternoon';
        else if (hour >= 18) greeting = 'Good evening';
        
        greetingEl.textContent = greetingEl.textContent.replace('Good morning', greeting);
    }
    
    // Auto-dismiss flash messages
    document.querySelectorAll('.flash-message').forEach(el => {
        setTimeout(() => {
            el.style.opacity = '0';
            setTimeout(() => el.remove(), 300);
        }, 4000);
    });
    
    // Add CSRF protection if needed
    const csrfMeta = document.querySelector('meta[name="csrf-token"]');
    if (csrfMeta) {
        const token = csrfMeta.content;
        // Add to all fetch requests
        const originalFetch = window.fetch;
        window.fetch = function(url, options = {}) {
            if (!options.headers) options.headers = {};
            options.headers['X-CSRFToken'] = token;
            return originalFetch(url, options);
        };
    }
});

// ========================================
// Export for use in other scripts
// ========================================

window.LUMINE = {
    debounce,
    formatDate,
    formatTime,
    getStatusColor,
    getStatusLabel,
    showToast,
    apiCall,
    validateEmail,
    validatePhone,
    formatInput,
    createTrendChart
};