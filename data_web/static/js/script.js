// View toggle functionality
document.addEventListener('DOMContentLoaded', function() {
    const viewButtons = document.querySelectorAll('.btn-view');
    const fileBrowser = document.getElementById('fileBrowser');
    
    // Load saved view preference
    const savedView = localStorage.getItem('viewMode') || 'grid';
    setView(savedView);
    
    viewButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            const view = this.dataset.view;
            setView(view);
            localStorage.setItem('viewMode', view);
        });
    });
    
    function setView(view) {
        viewButtons.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.view === view);
        });
        
        if (fileBrowser) {
            fileBrowser.classList.remove('grid-view', 'list-view');
            fileBrowser.classList.add(view + '-view');
        }
    }
    
    // Lazy load images
    if ('IntersectionObserver' in window) {
        const imageObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    if (img.dataset.src) {
                        img.src = img.dataset.src;
                        img.removeAttribute('data-src');
                    }
                    observer.unobserve(img);
                }
            });
        });
        
        document.querySelectorAll('img[data-src]').forEach(img => {
            imageObserver.observe(img);
        });
    }
    
    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Press 'g' to switch to grid view
        if (e.key === 'g' && !e.ctrlKey && !e.metaKey) {
            const gridBtn = document.querySelector('[data-view="grid"]');
            if (gridBtn) gridBtn.click();
        }
        
        // Press 'l' to switch to list view
        if (e.key === 'l' && !e.ctrlKey && !e.metaKey) {
            const listBtn = document.querySelector('[data-view="list"]');
            if (listBtn) listBtn.click();
        }
    });
});

// Image preview modal (optional enhancement)
function createLightbox() {
    const lightbox = document.createElement('div');
    lightbox.className = 'lightbox';
    lightbox.innerHTML = `
        <div class="lightbox-content">
            <button class="lightbox-close">&times;</button>
            <img src="" alt="">
        </div>
    `;
    document.body.appendChild(lightbox);
    
    lightbox.addEventListener('click', function(e) {
        if (e.target === lightbox || e.target.classList.contains('lightbox-close')) {
            lightbox.classList.remove('active');
        }
    });
    
    return lightbox;
}

// Drag and drop highlight (for future upload feature)
const fileBrowser = document.getElementById('fileBrowser');
if (fileBrowser) {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        fileBrowser.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
}

// Toast notifications
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    
    const container = document.querySelector('.toast-container') || createToastContainer();
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.classList.add('show');
    }, 10);
    
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function createToastContainer() {
    const container = document.createElement('div');
    container.className = 'toast-container';
    document.body.appendChild(container);
    return container;
}
