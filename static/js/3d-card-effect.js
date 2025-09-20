/**
 * 3D Card Tilt Effect
 * Creates an immersive 3D tilt effect based on mouse position
 */

document.addEventListener('DOMContentLoaded', function() {
    // Variables for 3D card effect
    const loginCard = document.getElementById('loginCard');
    if (!loginCard) return;
    
    // Create a container for the particles
    createParticles();
    
    // Maximum rotation angle in degrees
    const maxRotation = 10;
    
    // Track if mouse is over the card
    let isMouseOver = false;
    
    // Add event listeners
    loginCard.addEventListener('mousemove', handleMouseMove);
    loginCard.addEventListener('mouseenter', handleMouseEnter);
    loginCard.addEventListener('mouseleave', handleMouseLeave);
    
    /**
     * Handle mouse movement over the card
     */
    function handleMouseMove(event) {
        if (!isMouseOver) return;
        
        // Get card dimensions and position
        const rect = loginCard.getBoundingClientRect();
        
        // Calculate mouse position relative to the card
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        // Calculate rotation angles
        const rotateY = ((x / rect.width) * 2 - 1) * maxRotation;
        const rotateX = ((y / rect.height) * 2 - 1) * maxRotation * -1; // Invert Y axis
        
        // Apply rotation via CSS transform
        loginCard.style.transform = `
            perspective(1000px) 
            rotateX(${rotateX}deg) 
            rotateY(${rotateY}deg)
            translateZ(10px)
        `;
        
        // Update highlight effect
        updateHighlight(x / rect.width, y / rect.height);
    }
    
    /**
     * Handle mouse entering the card
     */
    function handleMouseEnter() {
        isMouseOver = true;
        
        // Add class for animation effect
        loginCard.classList.add('card-hover');
    }
    
    /**
     * Handle mouse leaving the card
     */
    function handleMouseLeave() {
        isMouseOver = false;
        
        // Reset transform
        loginCard.style.transform = `
            perspective(1000px) 
            rotateX(0deg) 
            rotateY(0deg)
            translateZ(0)
        `;
        
        // Reset highlight effect
        resetHighlight();
        
        // Remove hover class
        loginCard.classList.remove('card-hover');
    }
    
    /**
     * Update highlight effect based on mouse position
     */
    function updateHighlight(xPercent, yPercent) {
        // Create a radial gradient highlight that follows the cursor
        loginCard.style.backgroundImage = `
            radial-gradient(
                circle at ${xPercent * 100}% ${yPercent * 100}%,
                rgba(255, 255, 255, 0.1) 0%,
                rgba(255, 255, 255, 0.05) 20%,
                rgba(255, 255, 255, 0) 50%
            )
        `;
    }
    
    /**
     * Reset highlight effect
     */
    function resetHighlight() {
        loginCard.style.backgroundImage = '';
    }
    
    /**
     * Create floating particles behind the card
     */
    function createParticles() {
        // Create a container for particles
        const particleContainer = document.createElement('div');
        particleContainer.className = 'card-particles';
        particleContainer.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            pointer-events: none;
            overflow: hidden;
        `;
        
        loginCard.parentNode.insertBefore(particleContainer, loginCard);
        
        // Create particles
        for (let i = 0; i < 20; i++) {
            createParticle(particleContainer);
        }
        
        // Add mouseover event to create more particles
        loginCard.addEventListener('mouseover', function() {
            for (let i = 0; i < 10; i++) {
                setTimeout(() => {
                    createParticle(particleContainer);
                }, i * 100);
            }
        });
    }
    
    /**
     * Create a single particle
     */
    function createParticle(container) {
        const particle = document.createElement('div');
        
        // Set random position
        const x = Math.random() * 100;
        const y = Math.random() * 100;
        const size = Math.random() * 10 + 5;
        
        // Set random color
        const colors = ['rgba(255, 0, 128, 0.2)', 'rgba(37, 117, 252, 0.2)', 'rgba(255, 255, 255, 0.1)'];
        const color = colors[Math.floor(Math.random() * colors.length)];
        
        // Set random duration
        const duration = Math.random() * 10 + 5;
        const delay = Math.random() * 5;
        
        // Set CSS
        particle.style.cssText = `
            position: absolute;
            top: ${y}%;
            left: ${x}%;
            width: ${size}px;
            height: ${size}px;
            background: ${color};
            border-radius: 50%;
            filter: blur(2px);
            pointer-events: none;
            animation: float ${duration}s ease-in-out ${delay}s infinite;
            opacity: 0;
            transform: translateZ(0);
        `;
        
        // Add to container
        container.appendChild(particle);
        
        // Remove after animation completes
        setTimeout(() => {
            if (particle.parentNode === container) {
                container.removeChild(particle);
            }
        }, (duration + delay) * 1000);
    }
});